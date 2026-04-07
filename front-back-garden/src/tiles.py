"""
Multi-Source Tile Fetcher Module

Fetches aerial/satellite imagery tiles from multiple sources:
1. Google Map Tiles API
2. Manna Tiles API (internal)

Includes caching support to avoid re-fetching tiles for the same area.
"""

import hashlib
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

import config

# Allow large images (3000m radius at zoom 19 = ~1.1B pixels).
# We control the input (our own cached tiles), so the decompression bomb
# check is not needed and would block legitimate large-area fetches.
Image.MAX_IMAGE_PIXELS = None

# Cache directory
CACHE_DIR = Path(config.OUTPUT_DIR) / "cache"

# Global session token cache (for Google)
_session_token = None
_session_expiry = None

# ---------------------------------------------------------------------------
# Graceful shutdown support
# ---------------------------------------------------------------------------
# A threading.Event checked inside the tile fetch loop.  When set, the loop
# stops early so Ctrl+C / POST /api/shutdown actually terminates the process
# instead of leaving threads spinning on HTTP requests.
_shutdown_event = threading.Event()


def request_shutdown():
    """Signal all tile-fetching loops to stop."""
    _shutdown_event.set()


def clear_shutdown():
    """Reset the shutdown flag (call on server start)."""
    _shutdown_event.clear()


def is_shutdown_requested() -> bool:
    """Check whether a shutdown has been requested."""
    return _shutdown_event.is_set()


class TileSource(Enum):
    """Available tile sources."""
    GOOGLE = "google"
    MANNA = "manna"
    AUTO = "auto"  # Automatically choose best available


# Default tile source
DEFAULT_TILE_SOURCE = TileSource.GOOGLE


def get_cache_key(center_lat: float, center_lon: float, radius_m: float, zoom: int) -> str:
    """
    Generate a unique cache key for a given area.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        zoom: Zoom level
        
    Returns:
        Cache key string (hash)
    """
    # Round coordinates to 6 decimal places to handle floating point variations
    key_data = f"{center_lat:.6f}_{center_lon:.6f}_{radius_m:.1f}_{zoom}"
    return hashlib.md5(key_data.encode()).hexdigest()[:12]


def get_cache_paths(cache_key: str) -> Tuple[Path, Path]:
    """
    Get the paths for cached image and metadata.
    
    Args:
        cache_key: The cache key
        
    Returns:
        Tuple of (image_path, metadata_path)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = CACHE_DIR / f"{cache_key}_image.png"
    metadata_path = CACHE_DIR / f"{cache_key}_metadata.json"
    return image_path, metadata_path


def load_from_cache(
    center_lat: float, 
    center_lon: float, 
    radius_m: float, 
    zoom: int
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Try to load image and metadata from cache.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        zoom: Zoom level
        
    Returns:
        Tuple of (image, metadata) or (None, None) if not cached
    """
    cache_key = get_cache_key(center_lat, center_lon, radius_m, zoom)
    image_path, metadata_path = get_cache_paths(cache_key)
    
    if image_path.exists() and metadata_path.exists():
        try:
            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            return image, metadata
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None, None
    
    return None, None


def save_to_cache(
    image: np.ndarray,
    metadata: dict,
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int
):
    """
    Save image and metadata to cache.
    
    Args:
        image: Image as numpy array
        metadata: Metadata dict
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        zoom: Zoom level
    """
    cache_key = get_cache_key(center_lat, center_lon, radius_m, zoom)
    image_path, metadata_path = get_cache_paths(cache_key)
    
    try:
        # Save image
        Image.fromarray(image).save(image_path)
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Failed to save to cache: {e}")


def clear_cache():
    """Clear all cached tiles."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared.")


# =============================================================================
# MANNA TILE SOURCE
# =============================================================================

def get_manna_tile_url() -> Optional[str]:
    """
    Get the Manna tiles API URL from config.
    
    Returns:
        Manna tiles URL or None if not configured
    """
    return getattr(config, 'MANNA_TILES_URL', None)


def fetch_manna_tile(
    tile_x: int,
    tile_y: int,
    zoom: int,
    http_session: Optional[requests.Session] = None
) -> Optional[Image.Image]:
    """
    Fetch a single tile from Manna Tiles API.
    
    Manna tiles use standard XYZ tile format.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        http_session: Optional requests session for connection pooling
        
    Returns:
        PIL Image or None if fetch failed
    """
    manna_url = get_manna_tile_url()
    if not manna_url:
        return None
    
    # Build URL - Manna uses standard {z}/{x}/{y} format
    url = manna_url.format(z=zoom, x=tile_x, y=tile_y)
    
    # Add API key if configured
    manna_api_key = getattr(config, 'MANNA_API_KEY', None)
    headers = {}
    if manna_api_key:
        headers['Authorization'] = f'Bearer {manna_api_key}'
    
    try:
        requester = http_session if http_session else requests
        response = requester.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB")
        
    except requests.RequestException:
        return None


def check_manna_tile_availability(
    center_lat: float,
    center_lon: float,
    zoom: int = 19
) -> bool:
    """
    Check if Manna tiles are available for a location.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Zoom level
        
    Returns:
        True if Manna tiles are available
    """
    if not get_manna_tile_url():
        return False
    
    tile_x, tile_y = lat_lon_to_tile(center_lat, center_lon, zoom)
    tile = fetch_manna_tile(tile_x, tile_y, zoom)
    return tile is not None


# =============================================================================
# GOOGLE TILE SOURCE
# =============================================================================

def create_map_tiles_session(api_key: str) -> str:
    """
    Create a session token for the Map Tiles API.
    
    The session token is required for all tile requests and defines
    the map type and other parameters.
    
    Args:
        api_key: Google Maps API key
        
    Returns:
        Session token string
        
    Raises:
        Exception if session creation fails
    """
    global _session_token
    
    url = "https://tile.googleapis.com/v1/createSession"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    # Request satellite imagery
    payload = {
        "mapType": "satellite",
        "language": "en-US",
        "region": "IE",  # Ireland
    }
    
    params = {
        "key": api_key,
    }
    
    try:
        response = requests.post(url, json=payload, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        _session_token = data.get("session")
        
        if not _session_token:
            raise Exception(f"No session token in response: {data}")
        
        print(f"✅ Map Tiles API session created successfully")
        return _session_token
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"Failed to create Map Tiles session: {e}"
        try:
            error_data = response.json()
            if "error" in error_data:
                error_msg += f"\n   Error: {error_data['error'].get('message', error_data['error'])}"
                error_msg += f"\n   Status: {error_data['error'].get('status', 'unknown')}"
        except:
            pass
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"Failed to create Map Tiles session: {e}")


def get_session_token() -> str:
    """
    Get or create a session token for the Map Tiles API.
    
    Returns:
        Session token string
    """
    global _session_token
    
    if _session_token is None:
        _session_token = create_map_tiles_session(config.GOOGLE_TILES_API_KEY)
    
    return _session_token


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """
    Convert latitude/longitude to tile coordinates at a given zoom level.
    
    Uses Web Mercator projection (EPSG:3857).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level (0-22)
        
    Returns:
        Tuple of (tile_x, tile_y)
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    tile_x = int((lon + 180.0) / 360.0 * n)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return tile_x, tile_y


def tile_to_lat_lon(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile coordinates back to latitude/longitude (top-left corner of tile).
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        
    Returns:
        Tuple of (latitude, longitude) for top-left corner
    """
    n = 2 ** zoom
    lon = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def meters_to_tiles(meters: float, lat: float, zoom: int) -> int:
    """
    Calculate how many tiles are needed to cover a distance in meters.
    
    Args:
        meters: Distance in meters
        lat: Latitude (affects tile size due to projection)
        zoom: Zoom level
        
    Returns:
        Number of tiles needed
    """
    # Meters per pixel at equator for this zoom level
    meters_per_pixel_equator = 156543.03392 / (2 ** zoom)
    
    # Adjust for latitude
    meters_per_pixel = meters_per_pixel_equator * math.cos(math.radians(lat))
    
    # Pixels needed
    pixels_needed = meters / meters_per_pixel
    
    # Tiles needed (round up)
    tiles_needed = math.ceil(pixels_needed / config.TILE_SIZE)
    
    return max(1, tiles_needed)


# Maximum image dimension (width or height) in pixels that we'll attempt to
# process.  Memory estimates for the full pipeline (HSV, masks, distance
# transforms):
#   ~8k  px (zoom 18, ~750m)  →  ~240 MB RGB  → ~2-3 GB peak  → fine on 8 GB
#   ~12k px (zoom 18, ~1.1km) →  ~430 MB RGB  → ~4-5 GB peak  → fine on 16 GB
#   ~17k px (zoom 18, ~3km)   →  ~880 MB RGB  → ~8-10 GB peak → needs 16 GB+
#   ~34k px (zoom 19, ~3km)   →  ~3.5 GB RGB  → OOM on most machines
MAX_IMAGE_PIXELS_SIDE = 18_000


def recommended_zoom(radius_m: float, lat: float, preferred_zoom: int = None) -> int:
    """
    Return the highest zoom level that keeps the stitched image within
    MAX_IMAGE_PIXELS_SIDE on each side.

    Args:
        radius_m: Radius in meters
        lat: Latitude (affects meters-per-pixel)
        preferred_zoom: Zoom to start from (default: config.ZOOM_LEVEL)

    Returns:
        Zoom level (may be lower than preferred if image would be too large)
    """
    if preferred_zoom is None:
        preferred_zoom = config.ZOOM_LEVEL

    for z in range(preferred_zoom, 14, -1):
        tiles_radius = meters_to_tiles(radius_m, lat, z)
        side_tiles = 2 * tiles_radius + 1
        side_px = side_tiles * config.TILE_SIZE
        if side_px <= MAX_IMAGE_PIXELS_SIDE:
            return z

    return 15  # minimum reasonable zoom


def get_tiles_for_radius(
    center_lat: float, 
    center_lon: float, 
    radius_m: float, 
    zoom: int
) -> Tuple[List[Tuple[int, int]], Tuple[int, int, int, int]]:
    """
    Get all tile coordinates needed to cover a circular area.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        zoom: Zoom level
        
    Returns:
        Tuple of:
        - List of (tile_x, tile_y) coordinates
        - Bounding box as (min_x, min_y, max_x, max_y)
    """
    center_x, center_y = lat_lon_to_tile(center_lat, center_lon, zoom)
    
    # Calculate tiles needed in each direction
    tiles_radius = meters_to_tiles(radius_m, center_lat, zoom)
    
    # Generate all tiles in the bounding box
    min_x = center_x - tiles_radius
    max_x = center_x + tiles_radius
    min_y = center_y - tiles_radius
    max_y = center_y + tiles_radius
    
    tiles = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tiles.append((x, y))
    
    return tiles, (min_x, min_y, max_x, max_y)


# Counter for tile fetch errors (reset per fetch_area_image call)
_tile_error_count = 0
_TILE_ERROR_LOG_LIMIT = 5  # Only log the first N errors to avoid spam


def fetch_tile(
    tile_x: int, 
    tile_y: int, 
    zoom: int, 
    session_token: str,
    http_session: Optional[requests.Session] = None
) -> Optional[Image.Image]:
    """
    Fetch a single tile from Google Map Tiles API.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        session_token: Map Tiles API session token
        http_session: Optional requests session for connection pooling
        
    Returns:
        PIL Image or None if fetch failed
    """
    global _tile_error_count

    if not config.GOOGLE_TILES_API_KEY:
        raise ValueError(
            "Google Tiles API key not configured!\n"
            "Please add your API key to config.py: GOOGLE_TILES_API_KEY = 'your-key-here'"
        )
    
    # Map Tiles API endpoint
    url = f"https://tile.googleapis.com/v1/2dtiles/{zoom}/{tile_x}/{tile_y}"
    
    params = {
        "session": session_token,
        "key": config.GOOGLE_TILES_API_KEY,
    }
    
    try:
        requester = http_session if http_session else requests
        response = requester.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB")
        
    except Exception as e:
        _tile_error_count += 1
        if _tile_error_count <= _TILE_ERROR_LOG_LIMIT:
            # Log HTTP status and body for diagnosable errors
            detail = ""
            if isinstance(e, requests.HTTPError) and hasattr(e, "response") and e.response is not None:
                detail = f" [HTTP {e.response.status_code}]"
                try:
                    body = e.response.text[:300]
                    detail += f" {body}"
                except Exception:
                    pass
            print(f"    ⚠️  Tile fetch error ({tile_x},{tile_y},z{zoom}): {e}{detail}")
            if _tile_error_count == _TILE_ERROR_LOG_LIMIT:
                print(f"    (suppressing further tile errors)")
        return None


def fetch_area_image(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int = None,
    show_progress: bool = True,
    use_cache: bool = True,
    tile_source: TileSource = None
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Fetch and stitch all tiles for an area into a single image.
    
    Supports multiple tile sources:
    - Google Map Tiles API
    - Manna Tiles API
    - Auto (tries Manna first, falls back to Google)
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude  
        radius_m: Radius in meters
        zoom: Zoom level (uses config default if None)
        show_progress: Show progress bar
        use_cache: Whether to use cached tiles if available
        tile_source: Tile source to use (default: AUTO)
        
    Returns:
        Tuple of:
        - Numpy array (H, W, 3) RGB image or None if failed
        - Metadata dict with bounds, tile info, etc.
    """
    global _tile_error_count
    _tile_error_count = 0  # Reset per-request error counter

    if zoom is None:
        zoom = config.ZOOM_LEVEL

    # Auto-reduce zoom if the resulting image would be too large for memory
    safe = recommended_zoom(radius_m, center_lat, zoom)
    if safe != zoom:
        if show_progress:
            print(f"⚠️  Zoom auto-reduced {zoom} → {safe} "
                  f"(image at zoom {zoom} would exceed ~{MAX_IMAGE_PIXELS_SIDE}px limit)")
        zoom = safe

    if tile_source is None:
        tile_source = DEFAULT_TILE_SOURCE
    
    # Check cache first
    if use_cache:
        cached_image, cached_metadata = load_from_cache(center_lat, center_lon, radius_m, zoom)
        if cached_image is not None:
            print(f"📦 Loaded from cache (zoom {zoom}, {cached_metadata['image_size'][0]}x{cached_metadata['image_size'][1]} pixels)")
            return cached_image, cached_metadata
    
    # Get tiles needed
    tiles, (min_x, min_y, max_x, max_y) = get_tiles_for_radius(
        center_lat, center_lon, radius_m, zoom
    )
    
    # Determine which tile source to use
    use_manna = False
    use_google = False
    
    if tile_source == TileSource.AUTO:
        # Try Manna first
        if check_manna_tile_availability(center_lat, center_lon, zoom):
            use_manna = True
            print(f"Using Manna tiles (better coverage for this area)")
        elif config.GOOGLE_TILES_API_KEY:
            use_google = True
            print(f"Using Google tiles (Manna not available)")
        else:
            print("No tile source available - using placeholder")
            return create_placeholder_image(center_lat, center_lon, radius_m, zoom)
    elif tile_source == TileSource.MANNA:
        if get_manna_tile_url():
            use_manna = True
        else:
            print("Manna tiles not configured - falling back to Google")
            use_google = True
    else:  # TileSource.GOOGLE
        use_google = True
    
    print(f"Fetching {len(tiles)} tiles at zoom {zoom}...")

    # Create session token for Google if needed
    session_token = None
    if use_google:
        try:
            session_token = get_session_token()
        except Exception as e:
            print(f"❌ {e}")
            if use_manna:
                print("Falling back to Manna tiles...")
                use_google = False
            else:
                print("\nTo fix this, ensure the Map Tiles API is enabled:")
                print("  1. Go to: https://console.cloud.google.com/apis/library")
                print("  2. Search for 'Map Tiles API'")
                print("  3. Click 'Enable'")
                print("\nFalling back to placeholder image...")
                return create_placeholder_image(center_lat, center_lon, radius_m, zoom)

    # Calculate output image size
    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1
    output_width = width_tiles * config.TILE_SIZE
    output_height = height_tiles * config.TILE_SIZE

    # Create output image
    output = Image.new("RGB", (output_width, output_height))

    failed_tiles = []
    successful_tiles = 0

    # Each worker gets its own requests.Session for thread safety.
    # 24 workers is safe for both Google and Manna CDN-backed tile APIs.
    _TILE_WORKERS = 24
    _thread_local = threading.local()

    def _get_session() -> requests.Session:
        if not hasattr(_thread_local, "session"):
            _thread_local.session = requests.Session()
        return _thread_local.session

    def _fetch_one(tile_xy):
        tile_x, tile_y = tile_xy
        if _shutdown_event.is_set():
            return tile_x, tile_y, None
        s = _get_session()
        img = None
        if use_manna:
            img = fetch_manna_tile(tile_x, tile_y, zoom, s)
        if img is None and (use_google or (use_manna and session_token)):
            tok = session_token
            if tok is None:
                try:
                    tok = get_session_token()
                except Exception:
                    pass
            if tok:
                img = fetch_tile(tile_x, tile_y, zoom, tok, s)
        return tile_x, tile_y, img

    if show_progress:
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=len(tiles), desc="Fetching tiles")
    else:
        pbar = None

    with ThreadPoolExecutor(max_workers=_TILE_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tiles}
        for fut in as_completed(futures):
            if _shutdown_event.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                print("\n⚠️  Shutdown requested – aborting tile fetch")
                break
            tile_x, tile_y, tile_img = fut.result()
            if tile_img:
                paste_x = (tile_x - min_x) * config.TILE_SIZE
                paste_y = (tile_y - min_y) * config.TILE_SIZE
                output.paste(tile_img, (paste_x, paste_y))
                successful_tiles += 1
            else:
                failed_tiles.append((tile_x, tile_y))
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    
    if failed_tiles:
        if successful_tiles == 0:
            print(f"❌ All {len(tiles)} tiles failed to fetch")
            print("   This might be a rate limit or API issue.")
        else:
            print(f"Warning: {len(failed_tiles)} tiles failed to fetch")
    else:
        source_name = "Manna" if use_manna else "Google"
        print(f"✅ Successfully fetched {successful_tiles} tiles from {source_name}")
    
    # Calculate geographic bounds
    top_lat, left_lon = tile_to_lat_lon(min_x, min_y, zoom)
    bottom_lat, right_lon = tile_to_lat_lon(max_x + 1, max_y + 1, zoom)
    
    metadata = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_m": radius_m,
        "zoom": zoom,
        "tile_bounds": [min_x, min_y, max_x, max_y],  # Use list for JSON serialization
        "geo_bounds": {
            "north": top_lat,
            "south": bottom_lat,
            "west": left_lon,
            "east": right_lon,
        },
        "image_size": [output_width, output_height],  # Use list for JSON serialization
        "tiles_fetched": successful_tiles,
        "tiles_failed": len(failed_tiles),
        "tile_source": "manna" if use_manna else "google",
    }
    
    image_array = np.array(output)
    
    # Save to cache for future use
    if use_cache and successful_tiles > 0:
        save_to_cache(image_array, metadata, center_lat, center_lon, radius_m, zoom)
        print(f"💾 Saved to cache for future runs")
    
    return image_array, metadata


def create_placeholder_image(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int = None
) -> Tuple[np.ndarray, dict]:
    """
    Create a placeholder image when API key is not available.
    Used for testing the pipeline without actual imagery.
    
    Returns a gray image with grid lines representing tiles.
    """
    if zoom is None:
        zoom = config.ZOOM_LEVEL
        
    tiles, (min_x, min_y, max_x, max_y) = get_tiles_for_radius(
        center_lat, center_lon, radius_m, zoom
    )
    
    width_tiles = max_x - min_x + 1
    height_tiles = max_y - min_y + 1
    output_width = width_tiles * config.TILE_SIZE
    output_height = height_tiles * config.TILE_SIZE
    
    # Create gray placeholder
    output = np.ones((output_height, output_width, 3), dtype=np.uint8) * 200
    
    # Add grid lines
    for i in range(width_tiles + 1):
        x = i * config.TILE_SIZE
        output[:, max(0, x-1):min(output_width, x+1), :] = 150
    
    for i in range(height_tiles + 1):
        y = i * config.TILE_SIZE
        output[max(0, y-1):min(output_height, y+1), :, :] = 150
    
    # Calculate geographic bounds
    top_lat, left_lon = tile_to_lat_lon(min_x, min_y, zoom)
    bottom_lat, right_lon = tile_to_lat_lon(max_x + 1, max_y + 1, zoom)
    
    metadata = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_m": radius_m,
        "zoom": zoom,
        "tile_bounds": (min_x, min_y, max_x, max_y),
        "geo_bounds": {
            "north": top_lat,
            "south": bottom_lat,
            "west": left_lon,
            "east": right_lon,
        },
        "image_size": (output_width, output_height),
        "tiles_fetched": 0,
        "tiles_failed": len(tiles),
        "is_placeholder": True,
    }
    
    print(f"Created placeholder image ({output_width}x{output_height})")
    print("Note: Add your Google Tiles API key to config.py for real imagery")
    
    return output, metadata
