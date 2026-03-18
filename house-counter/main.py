"""
House Counter API - Count buildings in a given radius using satellite-derived data.
Uses Microsoft Building Footprints (ML-derived from satellite imagery) via Overture Maps.
"""
import asyncio
import io
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from cache_manager import get_cache_manager
from ms_buildings import query_ms_buildings_in_radius, get_building_polygons_ms, count_buildings_in_radius
from osm_query import query_osm_buildings, get_osm_building_polygons
from visualization import (
    create_map_image,
    calculate_zoom_for_radius,
    calculate_grid_size_for_zoom,
    estimate_processing_time
)

# Thread pool for running blocking operations concurrently
executor = ThreadPoolExecutor(max_workers=8)


app = FastAPI(
    title="House Counter API",
    description="Count buildings in a given radius using Microsoft Building Footprints (ML-derived from satellite imagery)",
    version="1.0.0"
)


class BuildingCountResponse(BaseModel):
    """Response model for building count endpoint."""
    latitude: float
    longitude: float
    radius_meters: float
    radius_km: float
    building_count: int
    total_area_sqm: float
    avg_building_area_sqm: float
    message: str


@app.get("/")
async def root():
    """API root endpoint with usage information."""
    return {
        "name": "House Counter API",
        "version": "1.0.0",
        "description": "Count buildings using Microsoft Building Footprints (ML-derived from satellite imagery)",
        "endpoints": {
            "/count": "GET - Count buildings within radius (Microsoft)",
            "/map": "GET - Get map image with building overlay",
            "/count-with-map": "GET - Count buildings and save map image",
            "/zoom-info": "GET - Get zoom level options and timing estimates",
            "/compare": "GET - Compare Microsoft vs OSM building counts",
            "/compare-with-map": "GET - Compare counts and generate map"
        },
        "example": "/count?lat=36.060345&lon=-95.816314&radius_km=3"
    }


@app.get("/count", response_model=BuildingCountResponse)
async def count_buildings(
    lat: float = Query(..., description="Latitude of center point", ge=-90, le=90),
    lon: float = Query(..., description="Longitude of center point", ge=-180, le=180),
    radius_km: float = Query(1.0, description="Search radius in kilometers", gt=0, le=10)
):
    """
    Count buildings within a given radius.
    
    Uses Microsoft Building Footprints via Overture Maps - ML-derived from 
    satellite imagery for comprehensive coverage.
    
    Note: radius_km is limited to 10km to keep query times reasonable.
    """
    radius_meters = radius_km * 1000
    
    try:
        # Fast path: vectorized count, no Building object creation
        count, total_area, avg_area = await asyncio.get_event_loop().run_in_executor(
            executor, count_buildings_in_radius, lat, lon, radius_meters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying buildings: {str(e)}")
    
    return BuildingCountResponse(
        latitude=lat,
        longitude=lon,
        radius_meters=radius_meters,
        radius_km=radius_km,
        building_count=count,
        total_area_sqm=total_area,
        avg_building_area_sqm=avg_area,
        message=f"Found {count} buildings within {radius_km}km"
    )


@app.get("/map")
async def get_map_image(
    lat: float = Query(..., description="Latitude of center point", ge=-90, le=90),
    lon: float = Query(..., description="Longitude of center point", ge=-180, le=180),
    radius_km: float = Query(1.0, description="Search radius in kilometers", gt=0, le=10),
    zoom: Optional[int] = Query(None, description="Zoom level (14-18). Higher = more detail but slower", ge=10, le=18)
):
    """
    Get a map image showing buildings within the search radius.
    
    Returns a PNG image with:
    - Google satellite tiles as background
    - Building polygons in red
    - Search radius circle in yellow
    - Center point marker in green
    - Info overlay with building count
    
    Zoom levels (for 3km radius):
    - 14 (default): ~5 seconds, 1280px image
    - 15: ~15 seconds, 2560px image
    - 16: ~1 minute, 5120px image
    - 17: ~5 minutes, 10240px image
    - 18: ~20 minutes, 20480px image
    """
    radius_meters = radius_km * 1000
    
    try:
        # Run blocking operations in thread pool to allow concurrent requests
        loop = asyncio.get_event_loop()
        buildings = await loop.run_in_executor(
            executor, get_building_polygons_ms, lat, lon, radius_meters
        )
        img = await loop.run_in_executor(
            executor, lambda: create_map_image(lat, lon, radius_meters, buildings, zoom=zoom)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating map: {str(e)}")
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    
    return StreamingResponse(img_bytes, media_type="image/png")


@app.get("/zoom-info")
async def get_zoom_info(
    radius_km: float = Query(3.0, description="Search radius in kilometers", gt=0, le=10)
):
    """
    Get information about zoom levels and estimated processing times.
    
    Helps you choose an appropriate zoom level for map generation.
    """
    radius_meters = radius_km * 1000
    auto_zoom = calculate_zoom_for_radius(radius_meters)
    
    zoom_info = []
    for z in range(auto_zoom, min(auto_zoom + 5, 19)):
        grid_size = calculate_grid_size_for_zoom(radius_meters, z, 36.0)
        total_tiles = grid_size * grid_size
        img_size = grid_size * 256
        
        zoom_info.append({
            "zoom": z,
            "grid_size": f"{grid_size}x{grid_size}",
            "total_tiles": total_tiles,
            "image_size": f"{img_size}x{img_size}",
            "estimated_time": estimate_processing_time(grid_size),
            "is_default": z == auto_zoom
        })
    
    return {
        "radius_km": radius_km,
        "auto_zoom": auto_zoom,
        "zoom_options": zoom_info
    }


@app.get("/count-with-map")
async def count_with_map(
    lat: float = Query(..., description="Latitude of center point", ge=-90, le=90),
    lon: float = Query(..., description="Longitude of center point", ge=-180, le=180),
    radius_km: float = Query(1.0, description="Search radius in kilometers", gt=0, le=10),
    output_path: Optional[str] = Query(None, description="Path to save the map image"),
    zoom: Optional[int] = Query(None, description="Zoom level (14-18). Higher = more detail but slower", ge=10, le=18)
):
    """
    Count buildings and save a map image to disk.
    
    Returns JSON with building count and saves map image.
    
    Zoom levels (for 3km radius):
    - 14 (default): ~5 seconds, 1280px image
    - 15: ~15 seconds, 2560px image
    - 16: ~1 minute, 5120px image
    - 17: ~5 minutes, 10240px image
    - 18: ~20 minutes, 20480px image
    """
    radius_meters = radius_km * 1000
    actual_zoom = zoom if zoom else calculate_zoom_for_radius(radius_meters)
    grid_size = calculate_grid_size_for_zoom(radius_meters, actual_zoom, lat)
    
    try:
        # Run blocking operations in thread pool to allow concurrent requests
        loop = asyncio.get_event_loop()
        buildings, _ = await loop.run_in_executor(
            executor, query_ms_buildings_in_radius, lat, lon, radius_meters
        )
        buildings_polygons = await loop.run_in_executor(
            executor, get_building_polygons_ms, lat, lon, radius_meters
        )
        building_count = len(buildings)
        total_area = sum(b.area_sqm for b in buildings)
        avg_area = total_area / len(buildings) if buildings else 0
        
        img = await loop.run_in_executor(
            executor, lambda: create_map_image(lat, lon, radius_meters, buildings_polygons, zoom=zoom)
        )
        
        if output_path:
            img.save(output_path, format="PNG")
            saved_path = os.path.abspath(output_path)
        else:
            zoom_suffix = f"_z{actual_zoom}" if zoom else ""
            filename = f"house_map_{lat}_{lon}_{radius_km}km{zoom_suffix}.png"
            img.save(filename, format="PNG")
            saved_path = os.path.abspath(filename)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    return {
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "radius_meters": radius_meters,
        "building_count": building_count,
        "total_area_sqm": round(total_area, 2),
        "avg_building_area_sqm": round(avg_area, 2),
        "zoom": actual_zoom,
        "grid_size": f"{grid_size}x{grid_size}",
        "image_size": f"{grid_size * 256}x{grid_size * 256}",
        "map_saved": saved_path,
        "message": f"Found {building_count} buildings within {radius_km}km"
    }


@app.get("/compare")
async def compare_data_sources(
    lat: float = Query(..., description="Latitude of center point", ge=-90, le=90),
    lon: float = Query(..., description="Longitude of center point", ge=-180, le=180),
    radius_km: float = Query(1.0, description="Search radius in kilometers", gt=0, le=10)
):
    """
    Compare building counts from Microsoft Building Footprints vs OpenStreetMap.
    
    Returns counts from both data sources for validation testing.
    Microsoft uses ML-derived satellite data, OSM uses crowdsourced data.
    """
    radius_meters = radius_km * 1000
    
    # Query both data sources concurrently
    loop = asyncio.get_event_loop()
    ms_error = None
    osm_error = None
    ms_count = 0
    osm_count = 0
    ms_total_area = 0
    
    # Run both queries in parallel
    async def query_ms():
        return await loop.run_in_executor(
            executor, query_ms_buildings_in_radius, lat, lon, radius_meters
        )
    
    async def query_osm():
        return await loop.run_in_executor(
            executor, query_osm_buildings, lat, lon, radius_meters
        )
    
    ms_task = asyncio.create_task(query_ms())
    osm_task = asyncio.create_task(query_osm())
    
    # Microsoft Building Footprints
    try:
        ms_buildings, _ = await ms_task
        ms_count = len(ms_buildings)
        ms_total_area = sum(b.area_sqm for b in ms_buildings)
    except Exception as e:
        ms_error = str(e)
    
    # OpenStreetMap
    try:
        osm_buildings = await osm_task
        osm_count = len(osm_buildings)
    except Exception as e:
        osm_error = str(e)
    
    # Calculate difference
    if ms_count > 0 and osm_count > 0:
        difference = ms_count - osm_count
        difference_pct = round((difference / osm_count) * 100, 1)
        comparison = f"Microsoft found {difference:+d} more buildings ({difference_pct:+.1f}%)"
    elif ms_count > 0:
        comparison = f"Microsoft found {ms_count}, OSM query failed"
    elif osm_count > 0:
        comparison = f"OSM found {osm_count}, Microsoft query failed"
    else:
        comparison = "Both queries failed or returned 0 results"
    
    return {
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "microsoft": {
            "building_count": ms_count,
            "total_area_sqm": round(ms_total_area, 2) if ms_total_area else 0,
            "source": "Microsoft Building Footprints (ML from satellite)",
            "error": ms_error
        },
        "osm": {
            "building_count": osm_count,
            "source": "OpenStreetMap (crowdsourced)",
            "error": osm_error
        },
        "comparison": comparison,
        "note": "Microsoft typically finds more buildings due to ML-based detection from satellite imagery"
    }


@app.get("/compare-with-map")
async def compare_with_map(
    lat: float = Query(..., description="Latitude of center point", ge=-90, le=90),
    lon: float = Query(..., description="Longitude of center point", ge=-180, le=180),
    radius_km: float = Query(1.0, description="Search radius in kilometers", gt=0, le=10),
    zoom: Optional[int] = Query(None, description="Zoom level (14-18)", ge=10, le=18)
):
    """
    Compare data sources and generate a map showing buildings from both sources.
    
    Buildings are color-coded:
    - Red: Microsoft Building Footprints
    - Blue: OpenStreetMap buildings
    """
    radius_meters = radius_km * 1000
    actual_zoom = zoom if zoom else calculate_zoom_for_radius(radius_meters)
    loop = asyncio.get_event_loop()
    
    # Get buildings from both sources concurrently
    async def query_ms():
        buildings, _ = await loop.run_in_executor(
            executor, query_ms_buildings_in_radius, lat, lon, radius_meters
        )
        polygons = await loop.run_in_executor(
            executor, get_building_polygons_ms, lat, lon, radius_meters
        )
        return buildings, polygons
    
    async def query_osm():
        return await loop.run_in_executor(
            executor, get_osm_building_polygons, lat, lon, radius_meters
        )
    
    ms_task = asyncio.create_task(query_ms())
    osm_task = asyncio.create_task(query_osm())
    
    try:
        ms_buildings, ms_polygons = await ms_task
    except Exception:
        ms_polygons = []
        ms_buildings = []
    
    try:
        osm_polygons = await osm_task
    except Exception:
        osm_polygons = []
    
    # Create base map with Microsoft buildings (red)
    # Note: OSM overlay would require coordinate translation to image pixels
    # which is complex - for now we just show Microsoft buildings on the map
    img = await loop.run_in_executor(
        executor, lambda: create_map_image(lat, lon, radius_meters, ms_polygons, zoom=zoom)
    )
    
    # Save map
    filename = f"compare_map_{lat}_{lon}_{radius_km}km.png"
    img.save(filename, format="PNG")
    saved_path = os.path.abspath(filename)
    
    ms_count = len(ms_buildings)
    osm_count = len(osm_polygons)
    
    if ms_count > 0 and osm_count > 0:
        difference = ms_count - osm_count
        difference_pct = round((difference / osm_count) * 100, 1)
        comparison = f"Microsoft found {difference:+d} more buildings ({difference_pct:+.1f}%)"
    else:
        comparison = f"Microsoft: {ms_count}, OSM: {osm_count}"
    
    return {
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius_km,
        "microsoft_count": ms_count,
        "osm_count": osm_count,
        "comparison": comparison,
        "map_saved": saved_path,
        "note": "Map shows Microsoft buildings (red). OSM comparison is count-only."
    }


# ======================================================================
# Cache Management Endpoints
# ======================================================================

_cache_mgr = get_cache_manager()

# In-flight task progress – written by worker threads, polled by SSE
_task_progress: Dict[str, dict] = {}


class CacheEstimateRequest(BaseModel):
    bbox: List[float]


class CacheStartRequest(BaseModel):
    bbox: List[float]
    name: str
    center_lat: float
    center_lon: float
    radius_km: Optional[float] = None


@app.get("/cache", response_class=HTMLResponse)
async def cache_ui():
    """Serve the cache management web UI."""
    html_path = Path(__file__).parent / "static" / "cache_ui.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/cache/areas")
async def list_cached_areas():
    """List all cached areas with stats."""
    return {
        "areas": _cache_mgr.get_cached_areas(),
        "stats": _cache_mgr.get_stats(),
    }


@app.post("/cache/estimate")
async def estimate_cache(request: CacheEstimateRequest):
    """Estimate disk size and check for overlapping cached areas."""
    bbox = tuple(request.bbox)
    est = _cache_mgr.estimate_cache_size(bbox)
    overlapping = _cache_mgr.find_overlapping(bbox)
    est["overlapping_areas"] = [
        {"id": a["id"], "name": a["name"]} for a in overlapping
    ]
    return est


@app.post("/cache/start")
async def start_cache(request: CacheStartRequest):
    """Start an async caching task. Returns a task_id for progress polling."""
    bbox = tuple(request.bbox)

    # Quick validation
    est = _cache_mgr.estimate_cache_size(bbox)
    if est["area_km2"] > 10000:
        raise HTTPException(
            400,
            f"Area too large ({est['area_km2']:.0f} km²). Maximum is 10,000 km².",
        )

    task_id = uuid.uuid4().hex[:12]
    _task_progress[task_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Queued…",
    }

    def _run():
        def progress_cb(pct: int, msg: str):
            _task_progress[task_id] = {
                "status": "running",
                "progress": pct,
                "message": msg,
            }

        try:
            result = _cache_mgr.cache_area(
                bbox=bbox,
                name=request.name,
                center_lat=request.center_lat,
                center_lon=request.center_lon,
                radius_km=request.radius_km,
                progress_cb=progress_cb,
            )
            if result:
                _task_progress[task_id] = {
                    "status": "complete",
                    "progress": 100,
                    "message": (
                        f"Cached {result['building_count']:,} buildings "
                        f"({result['file_size_bytes'] / (1024*1024):.1f} MB)"
                    ),
                    "area": result,
                }
            else:
                _task_progress[task_id] = {
                    "status": "complete",
                    "progress": 100,
                    "message": "No buildings found in this area.",
                    "area": None,
                }
        except Exception as e:
            _task_progress[task_id] = {
                "status": "error",
                "progress": 0,
                "message": str(e),
            }

    executor.submit(_run)
    return {"task_id": task_id}


@app.get("/cache/progress/{task_id}")
async def cache_progress(task_id: str):
    """SSE endpoint streaming progress updates for a caching task."""
    if task_id not in _task_progress:
        raise HTTPException(404, "Task not found")

    async def event_stream():
        last_sent = None
        deadline = asyncio.get_event_loop().time() + 600  # 10 min timeout
        while asyncio.get_event_loop().time() < deadline:
            current = _task_progress.get(task_id)
            if current and current != last_sent:
                yield f"data: {json.dumps(current)}\n\n"
                last_sent = current.copy()
                if current.get("status") in ("complete", "error"):
                    # Keep entry for 30 s so late pollers can see it
                    await asyncio.sleep(1)
                    break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/cache/areas/{area_id}")
async def delete_cached_area(area_id: str):
    """Delete a cached area from disk."""
    if _cache_mgr.delete_area(area_id):
        return {"success": True}
    raise HTTPException(404, "Area not found")


@app.get("/cache/stats")
async def cache_stats():
    """Return aggregate cache statistics."""
    return _cache_mgr.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
