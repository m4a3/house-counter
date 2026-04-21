#!/usr/bin/env python3
"""
Garden Classification API

FastAPI application providing REST endpoints for:
1. Single address lookup (eircode/GPS) -> front/back garden pins
2. Batch lookup (GPS + radius) -> all pins in area
3. Point classification (GPS) -> "front_garden" or "back_garden"
4. Precomputation management for large areas

Optimizations:
- Async endpoints with thread pool for CPU-bound work
- Per-coordinate caching (not chunk-based)
- Concurrent request handling
- Small processing radius for single-point queries

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints:
    GET  /health                     - Health check
    POST /api/garden-pins            - Get pins for single address
    POST /api/garden-pins/batch      - Get all pins in radius
    POST /api/classify               - Classify a GPS coordinate
    POST /api/precompute             - Precompute a large area
    GET  /api/cache/stats            - Get cache statistics
    DELETE /api/cache                - Clear caches
    POST /api/shutdown               - Gracefully stop the server
"""

import asyncio
import gc
import json as _json
import os
import queue
import threading
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from enum import Enum
from functools import partial
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.tiles import TileSource, fetch_area_image, request_shutdown, clear_shutdown
from src.precompute import PrecomputeManager
from src.fast_classifier import FastGardenClassifier, get_fast_classifier
from src.osm import fetch_buildings

# =============================================================================
# Thread Pool for CPU-bound work
# =============================================================================

# Thread pool for running CPU-intensive classification work
# Keep low to limit memory: each task can hold large numpy arrays (~1-2 GB)
_executor = ThreadPoolExecutor(max_workers=2)

# Global semaphore: only one precompute job at a time to prevent OOM.
_precompute_semaphore = asyncio.Semaphore(1)


# =============================================================================
# Live job registry — allows clients to reconnect to in-progress precomputes
# =============================================================================

class _LiveJob:
    """Tracks a running precompute job so late-joining SSE clients can catch up."""

    def __init__(self, key: str, lat: float, lon: float, radius_m: float, tile_source: str):
        self.key        = key
        self.lat        = lat
        self.lon        = lon
        self.radius_m   = radius_m
        self.tile_source = tile_source
        self.started_at = time.time()
        self.messages: list[str] = []   # full history (for replay)
        self.done       = False
        self.result     = None           # dict on success, str on error
        self.is_error   = False
        self._lock      = threading.Lock()

    def push(self, msg: str):
        with self._lock:
            self.messages.append(msg)

    def finish(self, result: dict):
        with self._lock:
            self.result = result
            self.done   = True

    def fail(self, error: str):
        with self._lock:
            self.result   = error
            self.is_error = True
            self.done     = True

    def to_meta(self) -> dict:
        with self._lock:
            return {
                "key":           self.key,
                "lat":           self.lat,
                "lon":           self.lon,
                "radius_m":      self.radius_m,
                "tile_source":   self.tile_source,
                "started_at":    self.started_at,
                "elapsed_s":     round(time.time() - self.started_at, 1),
                "message_count": len(self.messages),
                "done":          self.done,
            }


# Active jobs: key → _LiveJob.  Cleaned up after completion + 10 min.
_live_jobs: dict[str, _LiveJob] = {}


def _live_job_key(lat: float, lon: float, radius_m: float) -> str:
    return f"{lat:.4f}_{lon:.4f}_{radius_m:.0f}"


async def _watch_live_job(job: _LiveJob):
    """
    Async generator that replays buffered messages then streams new ones.
    Polls the job state every 300 ms; sends a heartbeat every 5 s of silence.
    """
    idx            = 0
    last_heartbeat = asyncio.get_event_loop().time()

    while True:
        with job._lock:
            batch    = list(job.messages[idx:])
            idx     += len(batch)
            done     = job.done
            is_error = job.is_error
            result   = job.result

        for msg in batch:
            yield f"data: {_json.dumps({'progress': msg})}\n\n"

        if done:
            if is_error:
                yield f"data: {_json.dumps({'status': 'error', 'message': result})}\n\n"
            else:
                yield f"data: {_json.dumps({'status': 'completed', 'summary': result})}\n\n"
            break

        now = asyncio.get_event_loop().time()
        if not batch and now - last_heartbeat >= 5:
            yield f"data: {_json.dumps({'heartbeat': True})}\n\n"
            last_heartbeat = now

        await asyncio.sleep(0.3)


async def run_in_executor(func, *args, **kwargs):
    """Run a blocking function in the thread pool."""
    loop = asyncio.get_event_loop()
    if kwargs:
        func = partial(func, **kwargs)
    return await loop.run_in_executor(_executor, func, *args)


# =============================================================================
# API Models
# =============================================================================

class TileSourceEnum(str, Enum):
    """Tile source options."""
    google = "google"
    manna = "manna"
    auto = "auto"


class GardenPinRequest(BaseModel):
    """Request for garden pins at a single location."""
    lat: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    lon: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    coords: Optional[str] = Field(None, description="Lat,lon as a string e.g. '53.380365, -6.386601'")
    eircode: Optional[str] = Field(None, description="Irish Eircode (alternative to lat/lon)")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    skip_cache: bool = Field(False, description="Skip cache and recompute")
    generate_map: bool = Field(False, description="Generate a visualization map of the pins")


class GardenPinResponse(BaseModel):
    """Response with front and back garden pins."""
    front: Optional[Dict[str, Any]] = Field(None, description="Best front garden delivery pin")
    back: Optional[Dict[str, Any]] = Field(None, description="Best back garden delivery pin")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    map_url: Optional[str] = Field(None, description="URL to visualization map (if generated)")


class BatchPinRequest(BaseModel):
    """Request for all garden pins within a radius."""
    lat: Optional[float] = Field(None, ge=-90, le=90, description="Center latitude")
    lon: Optional[float] = Field(None, ge=-180, le=180, description="Center longitude")
    coords: Optional[str] = Field(None, description="Lat,lon as a string e.g. '53.3498, -6.2603'")
    radius_m: float = Field(500, ge=10, le=5000, description="Search radius in meters")
    garden_type: Optional[str] = Field(None, description="Filter by 'front' or 'back'")
    min_score: float = Field(0.0, ge=0, le=100, description="Minimum score threshold (0 to include no-garden entries)")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    generate_map: bool = Field(False, description="Generate visualization map")


class BatchPinResponse(BaseModel):
    """Response with all pins in the area."""
    pins: List[Dict[str, Any]] = Field(default_factory=list, description="List of delivery pins")
    count: int = Field(0, description="Number of pins returned")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    map_url: Optional[str] = Field(None, description="URL to visualization map (if generated)")


class ClassifyRequest(BaseModel):
    """Request to classify a GPS coordinate."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    skip_cache: bool = Field(False, description="Skip cache and recompute")


class ClassifyResponse(BaseModel):
    """Response with classification result."""
    lat: float
    lon: float
    classification: str = Field(..., description="'front_garden', 'back_garden', or other surface type")
    surface_type: Optional[str] = Field(None, description="Type of surface")
    score: Optional[float] = Field(None, description="Delivery suitability score (0-100)")
    distance_to_building_m: Optional[float] = Field(None, description="Distance to nearest building")


class PrecomputeRequest(BaseModel):
    """Request to precompute a large area."""
    lat: float = Field(..., ge=-90, le=90, description="Center latitude")
    lon: float = Field(..., ge=-180, le=180, description="Center longitude")
    radius_m: float = Field(1000, ge=100, le=50000, description="Radius in meters")
    tile_source: TileSourceEnum = Field(TileSourceEnum.auto, description="Tile source to use")
    skip_cache: bool = Field(False, description="Force recompute even if cached")


class PrecomputeResponse(BaseModel):
    """Response from precomputation request."""
    status: str = Field(..., description="'started', 'completed', or 'error'")
    message: str = Field(..., description="Status message")
    summary: Optional[Dict[str, Any]] = Field(None, description="Computation summary")


class CacheStatsResponse(BaseModel):
    """Response with cache statistics."""
    fast_cache_entries: int
    precompute_areas: int
    total_pins: int


# =============================================================================
# Eircode Geocoding (Async)
# =============================================================================

# Irish Eircode routing key to approximate coordinates
# These are center points for each routing area
EIRCODE_ROUTING_COORDS = {
    # Dublin
    "D01": (53.3498, -6.2603), "D02": (53.3382, -6.2591), "D03": (53.3601, -6.2294),
    "D04": (53.3244, -6.2297), "D05": (53.3697, -6.2131), "D06": (53.3209, -6.2654),
    "D6W": (53.3102, -6.2878), "D07": (53.3567, -6.2817), "D08": (53.3387, -6.2928),
    "D09": (53.3782, -6.2334), "D10": (53.3519, -6.3249), "D11": (53.3867, -6.2842),
    "D12": (53.3189, -6.3172), "D13": (53.3923, -6.1968), "D14": (53.2978, -6.2478),
    "D15": (53.3879, -6.3892), "D16": (53.2858, -6.2108), "D17": (53.4012, -6.1521),
    "D18": (53.2689, -6.1789), "D20": (53.3492, -6.3678), "D22": (53.3189, -6.3892),
    "D24": (53.2889, -6.3892),
    # Add more routing keys as needed
}


def _is_in_ireland(lat: float, lon: float) -> bool:
    return 51.3 <= lat <= 55.5 and -10.7 <= lon <= -5.3


def _routing_key_matches(lat: float, lon: float, routing_key: str) -> bool:
    """Validate that geocoded coordinates are plausibly near the routing key area."""
    expected = EIRCODE_ROUTING_COORDS.get(routing_key)
    if expected is None:
        return True
    exp_lat, exp_lon = expected
    # Routing key areas are roughly 5-10km across; reject if >15km away
    dlat = abs(lat - exp_lat) * 111_000
    dlon = abs(lon - exp_lon) * 111_000 * 0.6  # cos(53°) ≈ 0.6
    return (dlat**2 + dlon**2) ** 0.5 < 15_000


async def geocode_eircode(eircode: str) -> Optional[Dict[str, float]]:
    """
    Geocode an Irish Eircode to lat/lon coordinates.

    Strategy (stops at first success):
    1. Google Geocoding API
    2. Google Places API (Text Search) — often enabled even when Geocoding isn't
    3. Nominatim structured postalcode search
    4. Nominatim free-text with routing key validation
    5. Routing key lookup table (approximate, always available)

    Every result is validated against the routing key area to prevent
    cross-Dublin mis-matches (e.g. D15 eircode returning a D02 location).
    """
    import config

    eircode_clean = eircode.strip().upper().replace(" ", "")

    if len(eircode_clean) != 7:
        print(f"Invalid eircode format: {eircode} (should be 7 characters)")
        return None

    eircode_formatted = f"{eircode_clean[:3]} {eircode_clean[3:]}"
    routing_key = eircode_clean[:3]
    nominatim_headers = {"User-Agent": "MannaGardenClassifier/1.0"}
    google_api_key = getattr(config, "GOOGLE_TILES_API_KEY", None)

    def _make_result(lat, lon, display_name, source):
        return {"lat": lat, "lon": lon, "display_name": display_name, "source": source}

    async with httpx.AsyncClient(timeout=15) as client:

        # --- Strategy 1: Google Geocoding API ---
        if google_api_key:
            try:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={
                        "address": eircode_formatted,
                        "key": google_api_key,
                        "region": "ie",
                        "components": "country:IE|postal_code:" + eircode_clean,
                    },
                )
                data = resp.json()
                if data.get("status") == "OK" and data.get("results"):
                    loc = data["results"][0]["geometry"]["location"]
                    lat, lon = loc["lat"], loc["lng"]
                    if _is_in_ireland(lat, lon) and _routing_key_matches(lat, lon, routing_key):
                        print(f"Eircode {eircode_formatted} geocoded via Google Geocoding → ({lat:.6f}, {lon:.6f})")
                        return _make_result(lat, lon, data["results"][0].get("formatted_address", eircode_formatted), "google")
                elif data.get("status") == "REQUEST_DENIED":
                    print("Google Geocoding API not enabled — trying Places API next")
            except Exception as e:
                print(f"Google Geocoding error: {e}")

        # --- Strategy 2: Google Places API (Text Search) ---
        if google_api_key:
            try:
                resp = await client.get(
                    "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
                    params={
                        "input": f"{eircode_formatted}, Ireland",
                        "inputtype": "textquery",
                        "fields": "formatted_address,geometry,name",
                        "locationbias": "circle:50000@53.4,-6.6",
                        "key": google_api_key,
                    },
                )
                data = resp.json()
                if data.get("status") == "OK" and data.get("candidates"):
                    loc = data["candidates"][0]["geometry"]["location"]
                    lat, lon = loc["lat"], loc["lng"]
                    if _is_in_ireland(lat, lon) and _routing_key_matches(lat, lon, routing_key):
                        name = data["candidates"][0].get("formatted_address", eircode_formatted)
                        print(f"Eircode {eircode_formatted} geocoded via Google Places → ({lat:.6f}, {lon:.6f})")
                        return _make_result(lat, lon, name, "google_places")
                elif data.get("status") == "REQUEST_DENIED":
                    print("Google Places API also not enabled")
            except Exception as e:
                print(f"Google Places error: {e}")

        # --- Strategy 3: Nominatim structured postalcode search ---
        try:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "postalcode": eircode_formatted,
                    "country": "Ireland",
                    "format": "json",
                    "limit": 5,
                    "addressdetails": 1,
                },
                headers=nominatim_headers,
            )
            for result in resp.json():
                lat, lon = float(result["lat"]), float(result["lon"])
                if _is_in_ireland(lat, lon) and _routing_key_matches(lat, lon, routing_key):
                    print(f"Eircode {eircode_formatted} geocoded via Nominatim (postalcode) → ({lat:.6f}, {lon:.6f})")
                    return _make_result(lat, lon, result.get("display_name", ""), "nominatim")
        except Exception as e:
            print(f"Nominatim postalcode lookup failed: {e}")

        # --- Strategy 4: Nominatim free-text with routing key constraint ---
        try:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": eircode_formatted,
                    "format": "json",
                    "countrycodes": "ie",
                    "limit": 10,
                    "addressdetails": 1,
                },
                headers=nominatim_headers,
            )
            for result in resp.json():
                lat, lon = float(result["lat"]), float(result["lon"])
                if _is_in_ireland(lat, lon) and _routing_key_matches(lat, lon, routing_key):
                    print(f"Eircode {eircode_formatted} geocoded via Nominatim (text) → ({lat:.6f}, {lon:.6f})")
                    return _make_result(lat, lon, result.get("display_name", ""), "nominatim")
            if resp.json():
                print(
                    f"Nominatim returned results for {eircode_formatted} but none matched "
                    f"routing key {routing_key} area — skipping bad matches"
                )
        except Exception as e:
            print(f"Nominatim text lookup failed: {e}")

    # --- Strategy 5: Routing key approximation (always works for known keys) ---
    if routing_key in EIRCODE_ROUTING_COORDS:
        lat, lon = EIRCODE_ROUTING_COORDS[routing_key]
        print(f"Eircode {eircode_formatted}: all geocoding APIs failed, using routing key {routing_key} approximation → ({lat:.6f}, {lon:.6f})")
        apis_needed = []
        if google_api_key:
            apis_needed.append("Enable Google Geocoding API: https://console.cloud.google.com/apis/library/geocoding-backend.googleapis.com")
            apis_needed.append("Or enable Places API: https://console.cloud.google.com/apis/library/places-backend.googleapis.com")
        else:
            apis_needed.append("Set GOOGLE_TILES_API_KEY in config and enable Geocoding API")
        for msg in apis_needed:
            print(f"  → {msg}")
        return {
            "lat": lat,
            "lon": lon,
            "display_name": f"{routing_key} area, Ireland (approximate)",
            "source": "routing_key",
            "approximate": True,
        }

    print(f"Could not geocode eircode: {eircode}")
    return None


# =============================================================================
# Lifecycle & signal handling
# =============================================================================

_signal_count = 0


def _handle_signal(signum, frame):
    """
    Handle SIGINT/SIGTERM:
    - 1st signal: set the shutdown flag so Python-level loops abort, then
      re-raise the default handler so uvicorn can shut down normally.
    - 2nd signal: force-kill immediately with os._exit().  This is needed
      when a worker is stuck inside a long-running C extension (e.g. OpenCV
      on a huge image) that never returns to Python to check the flag.
    """
    global _signal_count
    _signal_count += 1
    request_shutdown()

    if _signal_count >= 2:
        print(f"\n⚠️  Force-killing process (signal {signum}, attempt {_signal_count})...")
        os._exit(1)

    print(f"\n⚠️  Signal {signum} received – shutting down (press Ctrl+C again to force-kill)...")
    # Restore the default handler and re-raise so uvicorn handles shutdown
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the app."""
    global _signal_count
    # Startup: reset state and install signal handlers
    _signal_count = 0
    clear_shutdown()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    yield
    # Shutdown: ensure flag is set so any in-flight tile loops exit
    request_shutdown()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Garden Classification API",
    description="API for classifying front/back gardens and finding optimal delivery locations",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for serving generated maps
STATIC_DIR = Path(config.OUTPUT_DIR) / "maps"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_FRONTEND = Path(__file__).parent / "index.html"

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the garden classifier frontend."""
    return FileResponse(str(_FRONTEND), media_type="text/html")


def generate_pins_map(
    pins: List[Dict[str, Any]],
    center_lat: float,
    center_lon: float,
    radius_m: float
) -> Optional[str]:
    """
    Generate a visualization map showing buildings with garden pins.
    
    Returns the filename of the generated map.
    """
    import cv2
    import numpy as np
    from datetime import datetime
    
    try:
        # Fetch satellite imagery for the area
        image, metadata = fetch_area_image(
            center_lat, center_lon,
            radius_m=radius_m,
            zoom=19,
            tile_source=TileSource.AUTO,
            show_progress=False
        )
        
        if image is None:
            return None
        
        # Make a copy for drawing
        map_image = image.copy()
        geo_bounds = metadata["geo_bounds"]
        height, width = image.shape[:2]
        
        # Fetch buildings to draw
        buildings = fetch_buildings(center_lat, center_lon, radius_m, show_progress=False)
        
        # Draw building outlines
        for _, building in buildings.iterrows():
            try:
                if building.geometry.geom_type == 'Polygon':
                    coords = list(building.geometry.exterior.coords)
                elif building.geometry.geom_type == 'MultiPolygon':
                    coords = list(building.geometry.geoms[0].exterior.coords)
                else:
                    continue
                
                # Convert to pixel coordinates
                pts = []
                for lon, lat in coords:
                    px = int((lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * width)
                    py = int((geo_bounds["north"] - lat) / (geo_bounds["north"] - geo_bounds["south"]) * height)
                    pts.append([px, py])
                
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(map_image, [pts], True, (255, 255, 255), 2)  # White outline
            except Exception:
                continue
        
        # Score-to-color gradient (BGR)
        def score_to_color(score, garden_type):
            """Map score to BGR color. Uses green tones for front, blue tones for back."""
            if score <= 0:
                return (128, 128, 128)  # Gray for no garden
            
            if garden_type == "front":
                if score < 40:
                    return (0, 0, 200)      # Red (poor)
                elif score < 60:
                    return (0, 128, 255)    # Orange
                elif score < 75:
                    return (0, 200, 200)    # Yellow-green
                elif score < 90:
                    return (0, 220, 100)    # Lime
                else:
                    return (0, 255, 0)      # Bright green (excellent)
            else:  # back
                if score < 40:
                    return (100, 0, 200)    # Red-purple (poor)
                elif score < 60:
                    return (200, 80, 80)    # Muted blue
                elif score < 75:
                    return (220, 160, 0)    # Teal
                elif score < 90:
                    return (230, 180, 0)    # Light blue
                else:
                    return (255, 100, 0)    # Bright blue (excellent)
        
        # Draw pins with score-based colors
        for pin in pins:
            lat = pin["lat"]
            lon = pin["lon"]
            garden_type = pin.get("garden_type", "unknown")
            score = pin.get("score", 0)
            surface_type = pin.get("surface_type", "unknown")
            
            # Convert to pixel coordinates
            px = int((lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * width)
            py = int((geo_bounds["north"] - lat) / (geo_bounds["north"] - geo_bounds["south"]) * height)
            
            color = score_to_color(score, garden_type)
            
            if surface_type == "no_garden":
                # Draw distinctive marker at expected garden position
                # Orange X for front, purple X for back - easy to verify
                ng_color = (0, 165, 255) if garden_type == "front" else (200, 100, 200)
                size = 5
                cv2.line(map_image, (px-size, py-size), (px+size, py+size), ng_color, 2)
                cv2.line(map_image, (px-size, py+size), (px+size, py-size), ng_color, 2)
                # Label with "N" for no-garden
                cv2.putText(map_image, "N", (px + size + 2, py + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, ng_color, 1)
            else:
                # Size based on score (bigger = better)
                radius = max(4, int(score / 15))
                
                # Draw pin marker with score-based color
                cv2.circle(map_image, (px, py), radius, color, -1)
                cv2.circle(map_image, (px, py), radius + 1, (255, 255, 255), 1)
                
                # Surface type indicator + score label
                surface_prefix = {"grass": "G", "driveway": "D", "paved": "P", "tree": "T"}.get(surface_type, "?")
                label = f"{surface_prefix}{int(score)}"
                cv2.putText(map_image, label, (px + radius + 2, py + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Add legend with score ranges and surface types
        num_no_garden = sum(1 for p in pins if p.get("surface_type") == "no_garden")
        num_grass = sum(1 for p in pins if p.get("surface_type") == "grass")
        num_paved = sum(1 for p in pins if p.get("surface_type") in ("paved", "driveway"))
        num_with_garden = len(pins) - num_no_garden
        
        legend_h = 220
        cv2.rectangle(map_image, (10, 10), (260, 10 + legend_h), (0, 0, 0), -1)
        cv2.rectangle(map_image, (10, 10), (260, 10 + legend_h), (255, 255, 255), 1)
        
        y = 30
        cv2.putText(map_image, "Pin Labels: G=Grass D=Drive P=Paved", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 18
        cv2.putText(map_image, "Score Colors (front=green, back=blue):", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 18
        cv2.circle(map_image, (25, y), 5, (0, 255, 0), -1)
        cv2.putText(map_image, "90-100 (excellent grass)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 16
        cv2.circle(map_image, (25, y), 5, (0, 220, 100), -1)
        cv2.putText(map_image, "75-89 (good garden)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 16
        cv2.circle(map_image, (25, y), 5, (0, 200, 200), -1)
        cv2.putText(map_image, "60-74 (decent/driveway)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 16
        cv2.circle(map_image, (25, y), 5, (0, 128, 255), -1)
        cv2.putText(map_image, "40-59 (paved area)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 16
        cv2.circle(map_image, (25, y), 5, (0, 0, 200), -1)
        cv2.putText(map_image, "1-39 (poor)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        y += 18
        # No-garden markers
        sz = 4
        cv2.line(map_image, (22, y-2), (28, y+4), (0, 165, 255), 2)
        cv2.line(map_image, (22, y+4), (28, y-2), (0, 165, 255), 2)
        cv2.putText(map_image, "No front garden (expected pos)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
        y += 16
        cv2.line(map_image, (22, y-2), (28, y+4), (200, 100, 200), 2)
        cv2.line(map_image, (22, y+4), (28, y-2), (200, 100, 200), 2)
        cv2.putText(map_image, "No back garden (expected pos)", (40, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 100, 200), 1)
        y += 20
        cv2.putText(map_image, f"Grass: {num_grass} | Paved: {num_paved} | None: {num_no_garden}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
        
        # Save the map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pins_map_{timestamp}.png"
        filepath = STATIC_DIR / filename
        cv2.imwrite(str(filepath), map_image)
        
        # Free large image arrays
        del image, map_image
        gc.collect()
        
        return filename
        
    except Exception as e:
        print(f"Error generating map: {e}")
        return None


def get_tile_source(source_enum: TileSourceEnum) -> TileSource:
    """Convert API enum to internal TileSource."""
    return {
        TileSourceEnum.google: TileSource.GOOGLE,
        TileSourceEnum.manna: TileSource.MANNA,
        TileSourceEnum.auto: TileSource.AUTO
    }.get(source_enum, TileSource.AUTO)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    """Serve the garden classifier UI."""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.1.0"
    }


@app.post("/api/garden-pins", response_model=GardenPinResponse)
async def get_garden_pins(request: GardenPinRequest):
    """
    Get front and back garden delivery pins for a single location.
    
    Provide either:
    - lat/lon coordinates, OR
    - an Irish eircode
    
    Returns the best delivery pin locations for front and back gardens
    of the nearest house.
    
    Scoring:
    - 100 = grass (best)
    - 75 = driveway
    - 60 = paved area
    - 50 = potential parking area
    """
    start_time = time.time()
    
    # Resolve location
    lat, lon = None, None
    geocode_info = None
    
    if request.coords:
        try:
            parts = [p.strip() for p in request.coords.split(",")]
            lat, lon = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail=f"Invalid coords format: '{request.coords}'. Use 'lat, lon' e.g. '53.380365, -6.386601'")
    elif request.eircode:
        geocode_result = await geocode_eircode(request.eircode)
        if geocode_result is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not geocode eircode: {request.eircode}"
            )
        lat, lon = geocode_result["lat"], geocode_result["lon"]
        geocode_info = geocode_result.get("display_name")
    elif request.lat is not None and request.lon is not None:
        lat, lon = request.lat, request.lon
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'eircode' or both 'lat' and 'lon' must be provided"
        )
    
    # Get classifier
    tile_source = get_tile_source(request.tile_source)
    classifier = FastGardenClassifier(tile_source=tile_source)
    
    # Run classification in thread pool (CPU-bound)
    result = await run_in_executor(
        classifier.get_garden_pins,
        lat, lon,
        not request.skip_cache  # use_cache
    )
    
    elapsed = time.time() - start_time
    
    metadata = {
        "input_lat": lat,
        "input_lon": lon,
        "elapsed_seconds": round(elapsed, 3),
        "tile_source": request.tile_source.value,
        **result.get("metadata", {})
    }
    
    if request.eircode:
        metadata["eircode"] = request.eircode
        if geocode_info:
            metadata["geocoded_address"] = geocode_info
    
    map_url = None
    if request.generate_map:
        pins = [p for p in [result.get("front"), result.get("back")] if p]
        if pins:
            map_filename = await run_in_executor(
                generate_pins_map, pins, lat, lon, 50
            )
            if map_filename:
                map_url = f"/static/{map_filename}"
    
    return GardenPinResponse(
        front=result.get("front"),
        back=result.get("back"),
        metadata=metadata,
        map_url=map_url
    )


@app.post("/api/garden-pins/batch", response_model=BatchPinResponse)
async def get_garden_pins_batch(request: BatchPinRequest):
    """
    Get all garden delivery pins within a radius.
    
    Returns a list of front and back garden pins for all buildings
    within the specified radius.
    
    PERFORMANCE TIP: For large radii (>100m), precompute first:
        POST /api/precompute with same center and radius
    
    Then subsequent batch queries will be nearly instant (<1s).
    
    Set generate_map=true to create a visualization showing all pins.
    """
    start_time = time.time()
    
    lat, lon = request.lat, request.lon
    if request.coords:
        try:
            parts = [p.strip() for p in request.coords.split(",")]
            lat, lon = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail=f"Invalid coords format: '{request.coords}'. Use 'lat, lon' e.g. '53.3498, -6.2603'")
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Provide either 'lat'+'lon' or 'coords'")
    
    tile_source = get_tile_source(request.tile_source)
    manager = PrecomputeManager(tile_source=tile_source)
    
    # Check if area is precomputed
    is_cached = manager.is_area_cached(lat, lon, request.radius_m)
    
    # Acquire semaphore to prevent concurrent precomputes from OOMing
    async with _precompute_semaphore:
        pins = await run_in_executor(
            manager.get_pins_in_radius,
            lat,
            lon,
            request.radius_m,
            request.garden_type,
            request.min_score
        )
    
    elapsed = time.time() - start_time
    
    # Generate visualization map if requested
    map_url = None
    if request.generate_map and pins:
        map_filename = await run_in_executor(
            generate_pins_map,
            pins,
            lat,
            lon,
            request.radius_m
        )
        if map_filename:
            map_url = f"/static/{map_filename}"
    
    # Release memory from image/pin processing
    gc.collect()

    return BatchPinResponse(
        pins=pins,
        count=len(pins),
        metadata={
            "center_lat": lat,
            "center_lon": lon,
            "radius_m": request.radius_m,
            "garden_type": request.garden_type,
            "min_score": request.min_score,
            "elapsed_seconds": round(elapsed, 3),
            "tile_source": request.tile_source.value,
            "from_cache": is_cached,
            "tip": None if is_cached else "Precompute this area first for faster queries: POST /api/precompute"
        },
        map_url=map_url
    )


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_point(request: ClassifyRequest):
    """
    Classify a GPS coordinate as front garden, back garden, or other.
    
    Returns:
    - classification: "front_garden", "back_garden", "grass", "driveway", "building", "road", etc.
    - surface_type: The type of surface at this location
    - score: Delivery suitability score (0-100)
      - 100 = grass (best for delivery)
      - 75 = driveway
      - 60 = paved
      - 50 = parking area
    - distance_to_building_m: Distance to the nearest building
    """
    tile_source = get_tile_source(request.tile_source)
    classifier = FastGardenClassifier(tile_source=tile_source)
    
    # Run in thread pool
    result = await run_in_executor(
        classifier.classify_point,
        request.lat, request.lon,
        not request.skip_cache
    )
    
    return ClassifyResponse(
        lat=result.get("lat", request.lat),
        lon=result.get("lon", request.lon),
        classification=result.get("classification", "unknown"),
        surface_type=result.get("surface_type"),
        score=result.get("score"),
        distance_to_building_m=result.get("distance_to_building_m")
    )


@app.post("/api/precompute")
async def precompute_area_endpoint(request: PrecomputeRequest):
    """
    Precompute garden classification for a large area.

    Returns a stream of Server-Sent Events with progress updates.
    If the same area is already being computed (e.g. after a page refresh),
    the new connection replays all buffered messages and then streams the rest
    — no duplicate work, no data loss.
    """
    tile_source = get_tile_source(request.tile_source)
    manager = PrecomputeManager(tile_source=tile_source)

    if not request.skip_cache and manager.is_area_cached(request.lat, request.lon, request.radius_m):
        async def _cached():
            yield f"data: {_json.dumps({'status': 'completed', 'message': 'Already cached', 'summary': {'from_cache': True}})}\n\n"
        return StreamingResponse(_cached(), media_type="text/event-stream")

    job_key = _live_job_key(request.lat, request.lon, request.radius_m)

    # Purge stale completed jobs (> 10 min old)
    _now = time.time()
    for k in list(_live_jobs.keys()):
        j = _live_jobs[k]
        if j.done and _now - j.started_at > 600:
            _live_jobs.pop(k, None)

    # Reconnect to an already-running job (page refresh mid-precompute)
    existing = _live_jobs.get(job_key)
    if existing and not existing.done:
        async def _reconnect():
            yield f"data: {_json.dumps({'job_key': job_key, 'reconnected': True, 'elapsed_s': round(time.time() - existing.started_at, 1)})}\n\n"
            async for event in _watch_live_job(existing):
                yield event
        return StreamingResponse(_reconnect(), media_type="text/event-stream")

    # New job — register and start
    job = _LiveJob(job_key, request.lat, request.lon, request.radius_m,
                   request.tile_source or "auto")
    _live_jobs[job_key] = job

    def _progress_cb(msg: str):
        job.push(msg)

    def _run():
        try:
            summary = manager.precompute_area(
                request.lat, request.lon, request.radius_m,
                parallel=True, show_progress=True,
                progress_callback=_progress_cb,
            )
            job.finish(summary)
        except Exception as e:
            job.fail(str(e))
        finally:
            gc.collect()

    async def _stream():
        async with _precompute_semaphore:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(_executor, _run)
            # Send job key first so the client can persist it for resume
            yield f"data: {_json.dumps({'job_key': job_key, 'started': True})}\n\n"
            async for event in _watch_live_job(job):
                yield event

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/precompute/active")
async def get_active_precomputes():
    """Return metadata for all currently running precompute jobs."""
    active = [j.to_meta() for j in _live_jobs.values() if not j.done]
    return {"active_jobs": active}


@app.get("/api/precompute/check")
async def check_precompute_cached(lat: float, lon: float, radius_m: float):
    """
    Lightweight cache probe — returns whether an area is already precomputed.

    Never triggers a precompute. Safe to call on every map click or input change.
    """
    manager = PrecomputeManager()
    cached = manager.is_area_cached(lat, lon, radius_m)
    return {"cached": cached, "lat": lat, "lon": lon, "radius_m": radius_m}


@app.get("/api/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """Get statistics about the caches."""
    from pathlib import Path
    import config
    
    # Count fast cache entries
    fast_cache_dir = Path(config.OUTPUT_DIR) / "fast_cache"
    fast_cache_count = len(list(fast_cache_dir.glob("*.pkl"))) if fast_cache_dir.exists() else 0
    
    # Get precompute stats
    manager = PrecomputeManager()
    precompute_stats = manager.get_cache_stats()
    
    return CacheStatsResponse(
        fast_cache_entries=fast_cache_count,
        precompute_areas=precompute_stats.get("total_areas", 0),
        total_pins=precompute_stats.get("total_pins", 0)
    )


@app.delete("/api/cache")
async def clear_cache():
    """Clear all caches."""
    # Clear fast cache
    classifier = FastGardenClassifier()
    classifier.clear_cache()
    
    # Clear precompute cache
    manager = PrecomputeManager()
    manager.clear_cache()
    
    return {"status": "cleared", "message": "All caches cleared"}


@app.post("/api/shutdown")
async def shutdown_server():
    """
    Gracefully shut down the server.

    1. Sets the shutdown flag so in-flight tile fetches abort immediately.
    2. Schedules a SIGTERM to the current process so uvicorn exits cleanly.

    Usage:
        curl -X POST http://localhost:8000/api/shutdown
    """
    request_shutdown()

    # Give a moment for in-flight work to notice the flag, then kill ourselves
    async def _terminate():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.ensure_future(_terminate())
    return {"status": "shutting_down", "message": "Server is shutting down..."}


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Garden Classification API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (1 to avoid OOM on large areas)")
    
    args = parser.parse_args()
    
    print(f"\n🌿 Garden Classification API v1.1.0")
    print(f"   Starting server on http://{args.host}:{args.port}")
    print(f"   API docs: http://{args.host}:{args.port}/docs")
    print(f"   Workers: {args.workers}")
    print()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
