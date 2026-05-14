"""
Visualization module for creating map images with Google Tiles background.
"""
import io
import math
import os
import time
from typing import Callable, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import requests
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Platform-aware font discovery
_FONT_SEARCH_PATHS = [
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    # Linux (Liberation Sans — installed in Docker image)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    # Windows
    "C:/Windows/Fonts/arial.ttf",
]


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return the first available TrueType font at *size*, or the built-in default."""
    for path in _FONT_SEARCH_PATHS:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

# Load environment variables from .env file
load_dotenv()

# Google Maps tile URL
# If API key is available, use official endpoint; otherwise use public endpoint
GOOGLE_API_KEY = os.getenv("GOOGLE_TILES_API_KEY")

if GOOGLE_API_KEY:
    # Official Google Maps Tiles API
    GOOGLE_TILE_URL = f"https://tile.googleapis.com/v1/2dtiles/{{z}}/{{x}}/{{y}}?session=YOUR_SESSION&key={GOOGLE_API_KEY}"
    # For now, fall back to public endpoint with key as backup
    # The official Tiles API requires session tokens which adds complexity
    GOOGLE_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    print(f"Google API key loaded (using optimized endpoint)")
else:
    # Public endpoint (no API key)
    GOOGLE_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to lat/lon (top-left corner of tile)."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def get_tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """Get the lat/lon bounds of a tile (north, south, east, west)."""
    north, west = tile_to_lat_lon(x, y, zoom)
    south, east = tile_to_lat_lon(x + 1, y + 1, zoom)
    return north, south, east, west


def lat_lon_to_pixel(
    lat: float, 
    lon: float, 
    tile_x: int, 
    tile_y: int, 
    zoom: int,
    tile_size: int = 256
) -> Tuple[int, int]:
    """Convert lat/lon to pixel coordinates within a tile grid."""
    n = 2 ** zoom
    
    # Get pixel position in world coordinates
    world_x = (lon + 180.0) / 360.0 * n * tile_size
    lat_rad = math.radians(lat)
    world_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * tile_size
    
    # Get the pixel position relative to the tile grid origin
    tile_origin_x = tile_x * tile_size
    tile_origin_y = tile_y * tile_size
    
    pixel_x = int(world_x - tile_origin_x)
    pixel_y = int(world_y - tile_origin_y)
    
    return pixel_x, pixel_y


def calculate_zoom_for_radius(radius_meters: float) -> int:
    """Calculate appropriate zoom level for a given radius."""
    # Approximate meters per pixel at equator for each zoom level
    # At zoom 0, the entire world is 256 pixels, Earth's circumference ~40,075 km
    meters_per_pixel_at_equator = 40075016.686 / 256
    
    for zoom in range(20, 0, -1):
        meters_per_pixel = meters_per_pixel_at_equator / (2 ** zoom)
        # We want the radius to fit in about 1/3 of the image
        pixels_for_radius = radius_meters / meters_per_pixel
        if pixels_for_radius < 400:  # Target around 400 pixels for the radius
            return zoom
    return 15


def calculate_grid_size_for_zoom(
    radius_meters: float, 
    zoom: int, 
    lat: float,
    base_grid: int = 5
) -> int:
    """
    Calculate grid size needed to cover the radius at the given zoom level.
    
    At the auto-calculated zoom, we use base_grid (5).
    For higher zooms, we need more tiles to cover the same area.
    """
    auto_zoom = calculate_zoom_for_radius(radius_meters)
    
    if zoom <= auto_zoom:
        return base_grid
    
    # Each zoom level doubles the tiles needed
    zoom_diff = zoom - auto_zoom
    multiplier = 2 ** zoom_diff
    
    grid_size = base_grid * multiplier
    
    # Cap at reasonable maximum to prevent memory issues
    max_grid = 100  # 100x100 = 10,000 tiles max
    return min(grid_size, max_grid)


def estimate_processing_time(grid_size: int) -> str:
    """Estimate processing time based on grid size."""
    total_tiles = grid_size * grid_size
    # Assume ~150ms per tile with parallel downloads
    seconds = total_tiles * 0.05  # With 10 parallel workers
    
    if seconds < 60:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        return f"~{int(seconds/60)} minutes"
    else:
        return f"~{seconds/3600:.1f} hours"


def _building_union(buildings: Optional[List[dict]]):
    """Unary-union all buildings into a single shapely geometry.

    Input dicts have ``coordinates`` as ``[(lat, lon), ...]`` rings. Returns
    ``None`` if the input is empty or no valid polygons could be built.
    """
    if not buildings:
        return None
    polys = []
    for b in buildings:
        coords = b.get("coordinates", [])
        if len(coords) < 3:
            continue
        ring = [(lon, lat) for lat, lon in coords]
        try:
            p = Polygon(ring)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_valid and not p.is_empty:
                polys.append(p)
        except Exception:
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:
        return None


def _draw_geometry(
    region,
    draw: ImageDraw.ImageDraw,
    to_pixel: Callable[[float, float], Tuple[int, int]],
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
):
    """Rasterise a shapely Polygon/MultiPolygon/GeometryCollection."""
    if region is None or region.is_empty:
        return

    geoms = []
    if region.geom_type == "Polygon":
        geoms = [region]
    elif region.geom_type == "MultiPolygon":
        geoms = list(region.geoms)
    elif region.geom_type == "GeometryCollection":
        for g in region.geoms:
            if g.geom_type == "Polygon":
                geoms.append(g)
            elif g.geom_type == "MultiPolygon":
                geoms.extend(g.geoms)
    else:
        return

    for poly in geoms:
        exterior_px = [to_pixel(lat, lon) for lon, lat in poly.exterior.coords]
        if len(exterior_px) < 3:
            continue
        draw.polygon(exterior_px, fill=fill, outline=outline)
        # Building polygons rarely have interior rings; if they do, punch
        # the hole back through with a fully-transparent fill so the
        # underlying tile shows.
        for interior in poly.interiors:
            hole_px = [to_pixel(lat, lon) for lon, lat in interior.coords]
            if len(hole_px) >= 3:
                draw.polygon(hole_px, fill=(0, 0, 0, 0), outline=outline)


def fetch_tile(x: int, y: int, zoom: int) -> Tuple[int, int, Image.Image | None]:
    """Fetch a single Google Maps tile. Returns (x, y, image)."""
    url = GOOGLE_TILE_URL.format(x=x, y=y, z=zoom)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; HouseCounter/1.0)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return (x, y, Image.open(io.BytesIO(response.content)))
    except Exception as e:
        return (x, y, None)


def fetch_tiles_parallel(
    tiles: List[Tuple[int, int, int]], 
    max_workers: int = 10,
    progress_callback=None
) -> dict:
    """
    Fetch multiple tiles in parallel.
    
    Args:
        tiles: List of (x, y, zoom) tuples
        max_workers: Number of parallel download threads
        progress_callback: Optional callback(completed, total) for progress updates
    
    Returns:
        Dict mapping (x, y) to Image
    """
    results = {}
    total = len(tiles)
    completed = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_tile, x, y, z): (x, y) 
            for x, y, z in tiles
        }
        
        for future in as_completed(futures):
            x, y, img = future.result()
            if img:
                results[(x, y)] = img
            
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total)
            elif completed % 100 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  Downloaded {completed}/{total} tiles ({rate:.1f}/s, ETA: {eta:.0f}s)")
    
    return results


def create_map_image(
    center_lat: float,
    center_lon: float,
    radius_meters: float,
    buildings: List[dict],
    tile_size: int = 256,
    grid_size: int = 5,
    zoom: Optional[int] = None,
    osm_buildings: Optional[List[dict]] = None,
) -> Image.Image:
    """
    Create a map image with Google Tiles background and building markers.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_meters: Radius in meters
        buildings: Microsoft/Overture building dicts with 'coordinates' and 'center'.
            Drawn in red.
        tile_size: Size of each tile (default 256)
        grid_size: Number of tiles in each direction (default 5, auto-calculated if zoom is specified)
        zoom: Optional zoom level override (default: auto-calculated based on radius)
        osm_buildings: Optional second list of building dicts to overlay in blue.
            Same shape as ``buildings``. When provided, the info box and legend
            also show OSM coverage so blind spots (red-only areas) and
            shape mismatches (red vs. blue outlines) are easy to spot.

    Returns:
        PIL Image with map and overlays
    """
    # Calculate or use provided zoom
    if zoom is None:
        zoom = calculate_zoom_for_radius(radius_meters)
        actual_grid_size = grid_size
    else:
        # Calculate grid size needed for this zoom to cover the radius
        actual_grid_size = calculate_grid_size_for_zoom(radius_meters, zoom, center_lat, grid_size)
    
    total_tiles = actual_grid_size * actual_grid_size
    print(f"Generating map at zoom {zoom} with {actual_grid_size}x{actual_grid_size} = {total_tiles} tiles")
    print(f"Estimated time: {estimate_processing_time(actual_grid_size)}")
    
    # Get center tile
    center_tile_x, center_tile_y = lat_lon_to_tile(center_lat, center_lon, zoom)
    
    # Calculate tile range (centered around the center tile)
    half_grid = actual_grid_size // 2
    start_tile_x = center_tile_x - half_grid
    start_tile_y = center_tile_y - half_grid
    
    # Create composite image
    img_width = tile_size * actual_grid_size
    img_height = tile_size * actual_grid_size
    
    print(f"Output image size: {img_width}x{img_height} pixels")
    
    composite = Image.new("RGB", (img_width, img_height), (200, 200, 200))
    
    # Build list of tiles to fetch
    tiles_to_fetch = []
    for dy in range(actual_grid_size):
        for dx in range(actual_grid_size):
            tile_x = start_tile_x + dx
            tile_y = start_tile_y + dy
            tiles_to_fetch.append((tile_x, tile_y, zoom))
    
    # Fetch tiles in parallel
    print("Downloading tiles...")
    tile_images = fetch_tiles_parallel(tiles_to_fetch, max_workers=20)
    
    # Place tiles in composite
    print("Compositing tiles...")
    for dy in range(actual_grid_size):
        for dx in range(actual_grid_size):
            tile_x = start_tile_x + dx
            tile_y = start_tile_y + dy
            if (tile_x, tile_y) in tile_images:
                composite.paste(tile_images[(tile_x, tile_y)], (dx * tile_size, dy * tile_size))
    
    draw = ImageDraw.Draw(composite, "RGBA")
    
    # Draw radius circle
    center_px, center_py = lat_lon_to_pixel(
        center_lat, center_lon, 
        start_tile_x, start_tile_y, 
        zoom, tile_size
    )
    
    # Calculate pixel radius
    meters_per_pixel = 40075016.686 * math.cos(math.radians(center_lat)) / (256 * 2**zoom)
    radius_px = int(radius_meters / meters_per_pixel)
    
    # Draw search radius circle
    circle_width = max(3, actual_grid_size // 10)  # Scale circle width with image size
    draw.ellipse(
        [
            center_px - radius_px, center_py - radius_px,
            center_px + radius_px, center_py + radius_px
        ],
        outline=(255, 255, 0, 200),
        width=circle_width
    )
    
    osm_count = len(osm_buildings) if osm_buildings else 0

    def _to_pixel(lat: float, lon: float) -> Tuple[int, int]:
        return lat_lon_to_pixel(lat, lon, start_tile_x, start_tile_y, zoom, tile_size)

    if osm_buildings is not None:
        # Set-difference overlay: green where both sources agree, red where
        # only MS has a building (OSM blind spot), blue where only OSM has it.
        print(
            f"Computing overlap regions for {len(buildings)} MS vs "
            f"{osm_count} OSM polygons..."
        )
        t_union = time.time()
        ms_union = _building_union(buildings)
        osm_union = _building_union(osm_buildings)
        print(f"  unions built in {time.time() - t_union:.1f}s")

        if ms_union is not None and osm_union is not None:
            t_ops = time.time()
            try:
                agreement = ms_union.intersection(osm_union)
            except Exception:
                agreement = None
            try:
                ms_only = ms_union.difference(osm_union)
            except Exception:
                ms_only = ms_union
            try:
                osm_only = osm_union.difference(ms_union)
            except Exception:
                osm_only = osm_union
            print(f"  set operations done in {time.time() - t_ops:.1f}s")
        else:
            agreement = None
            ms_only = ms_union
            osm_only = osm_union

        # Draw agreement first so its outline isn't overwritten by the
        # red / blue diff regions (which are disjoint anyway).
        _draw_geometry(
            agreement, draw, _to_pixel,
            fill=(0, 255, 60, 140), outline=(0, 230, 30, 255),
        )
        _draw_geometry(
            ms_only, draw, _to_pixel,
            fill=(255, 0, 0, 140), outline=(255, 0, 0, 255),
        )
        _draw_geometry(
            osm_only, draw, _to_pixel,
            fill=(0, 150, 255, 140), outline=(0, 150, 255, 255),
        )
    else:
        # Single-source rendering (no OSM overlay) — preserve original red.
        print(f"Drawing {len(buildings)} building polygons...")
        for building in buildings:
            coords = building.get("coordinates", [])
            if len(coords) < 3:
                continue

            pixel_coords = []
            for lat, lon in coords:
                px, py = _to_pixel(lat, lon)
                pixel_coords.append((px, py))

            draw.polygon(
                pixel_coords,
                fill=(255, 0, 0, 100), outline=(255, 0, 0, 255),
            )
    
    # Draw center marker. When the OSM overlay is active, switch to magenta
    # so it can't be mistaken for the "both sources agree" green regions.
    marker_size = max(8, actual_grid_size // 5)
    if osm_buildings is not None:
        marker_fill = (255, 0, 255, 255)
        marker_outline = (120, 0, 120, 255)
    else:
        marker_fill = (0, 255, 0, 255)
        marker_outline = (0, 100, 0, 255)
    draw.ellipse(
        [
            center_px - marker_size, center_py - marker_size,
            center_px + marker_size, center_py + marker_size,
        ],
        fill=marker_fill,
        outline=marker_outline,
    )
    
    # Add text overlay with count
    # Scale font size with image
    font_size = max(24, img_width // 50)
    small_font_size = max(16, img_width // 75)
    font = _find_font(font_size)
    small_font = _find_font(small_font_size)
    
    # Create semi-transparent overlay for text
    overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Draw info box (scaled with image size) — taller when OSM is overlaid
    box_width = max(360, img_width // 4)
    info_lines = 3 if osm_buildings is None else 4
    box_height = max(120, img_height // 10) + (small_font_size + 6 if osm_buildings else 0)
    box_padding = max(15, img_width // 80)
    box_x, box_y = 20, 20

    overlay_draw.rectangle(
        [box_x, box_y, box_x + box_width, box_y + box_height],
        fill=(0, 0, 0, 180)
    )

    ms_label = "MS Buildings" if osm_buildings is not None else "Buildings Found"
    info_text = f"{ms_label}: {len(buildings)}"
    coord_text = f"Center: {center_lat:.6f}, {center_lon:.6f}"
    radius_text = f"Radius: {radius_meters/1000:.1f} km | Zoom: {zoom}"

    ty = box_y + box_padding
    overlay_draw.text((box_x + box_padding, ty), info_text, fill=(255, 255, 255), font=font)
    ty += font_size + 5
    if osm_buildings is not None:
        osm_text = f"OSM Buildings: {osm_count}"
        overlay_draw.text((box_x + box_padding, ty), osm_text, fill=(180, 220, 255), font=small_font)
        ty += small_font_size + 5
    overlay_draw.text((box_x + box_padding, ty), coord_text, fill=(200, 200, 200), font=small_font)
    ty += small_font_size + 5
    overlay_draw.text((box_x + box_padding, ty), radius_text, fill=(200, 200, 200), font=small_font)

    # Legend (scaled) — extra rows when both sources are overlaid
    legend_rows = 4 if osm_buildings is not None else 2
    legend_height = max(60, img_height // 15) + (
        (small_font_size + 6) * 2 if osm_buildings is not None else 0
    )
    legend_width = max(280, img_width // 5)
    legend_y = img_height - legend_height - 20
    overlay_draw.rectangle(
        [20, legend_y, 20 + legend_width, legend_y + legend_height],
        fill=(0, 0, 0, 180),
    )

    icon_size = max(10, legend_height // (legend_rows * 2))
    row_y = legend_y + icon_size

    if osm_buildings is not None:
        overlay_draw.rectangle(
            [35, row_y, 35 + icon_size, row_y + icon_size],
            fill=(0, 255, 60, 180), outline=(0, 230, 30),
        )
        overlay_draw.text(
            (50 + icon_size, row_y - 3),
            "Both sources agree", fill=(255, 255, 255), font=small_font,
        )
        row_y += icon_size * 2

        overlay_draw.rectangle(
            [35, row_y, 35 + icon_size, row_y + icon_size],
            fill=(255, 0, 0, 180), outline=(255, 0, 0),
        )
        overlay_draw.text(
            (50 + icon_size, row_y - 3),
            "MS / Overture only", fill=(255, 255, 255), font=small_font,
        )
        row_y += icon_size * 2

        overlay_draw.rectangle(
            [35, row_y, 35 + icon_size, row_y + icon_size],
            fill=(0, 150, 255, 180), outline=(0, 150, 255),
        )
        overlay_draw.text(
            (50 + icon_size, row_y - 3),
            "OSM only", fill=(255, 255, 255), font=small_font,
        )
        row_y += icon_size * 2
    else:
        overlay_draw.rectangle(
            [35, row_y, 35 + icon_size, row_y + icon_size],
            fill=(255, 0, 0, 150), outline=(255, 0, 0),
        )
        overlay_draw.text(
            (50 + icon_size, row_y - 3),
            "MS / Overture Building", fill=(255, 255, 255), font=small_font,
        )
        row_y += icon_size * 2

    if osm_buildings is not None:
        center_fill = (255, 0, 255)
        center_outline = (120, 0, 120)
    else:
        center_fill = (0, 255, 0)
        center_outline = (0, 100, 0)
    overlay_draw.ellipse(
        [35, row_y, 35 + icon_size, row_y + icon_size],
        fill=center_fill, outline=center_outline,
    )
    overlay_draw.text(
        (50 + icon_size, row_y - 3),
        "Center Point", fill=(255, 255, 255), font=small_font,
    )
    
    # Composite the overlay
    composite = Image.alpha_composite(composite.convert("RGBA"), overlay)
    
    print("Map generation complete!")
    return composite.convert("RGB")


def create_simple_marker_map(
    center_lat: float,
    center_lon: float,
    radius_meters: float,
    building_centers: List[Tuple[float, float]],
    tile_size: int = 256,
    grid_size: int = 5,
    zoom: Optional[int] = None
) -> Image.Image:
    """
    Create a simpler map with just markers for building centers (faster).
    """
    if zoom is None:
        zoom = calculate_zoom_for_radius(radius_meters)
        actual_grid_size = grid_size
    else:
        actual_grid_size = calculate_grid_size_for_zoom(radius_meters, zoom, center_lat, grid_size)
    
    center_tile_x, center_tile_y = lat_lon_to_tile(center_lat, center_lon, zoom)
    half_grid = actual_grid_size // 2
    start_tile_x = center_tile_x - half_grid
    start_tile_y = center_tile_y - half_grid
    
    img_width = tile_size * actual_grid_size
    img_height = tile_size * actual_grid_size
    composite = Image.new("RGB", (img_width, img_height), (200, 200, 200))
    
    # Build list of tiles to fetch
    tiles_to_fetch = []
    for dy in range(actual_grid_size):
        for dx in range(actual_grid_size):
            tile_x = start_tile_x + dx
            tile_y = start_tile_y + dy
            tiles_to_fetch.append((tile_x, tile_y, zoom))
    
    # Fetch tiles in parallel
    tile_images = fetch_tiles_parallel(tiles_to_fetch, max_workers=20)
    
    # Place tiles
    for dy in range(actual_grid_size):
        for dx in range(actual_grid_size):
            tile_x = start_tile_x + dx
            tile_y = start_tile_y + dy
            if (tile_x, tile_y) in tile_images:
                composite.paste(tile_images[(tile_x, tile_y)], (dx * tile_size, dy * tile_size))
    
    draw = ImageDraw.Draw(composite, "RGBA")
    
    center_px, center_py = lat_lon_to_pixel(
        center_lat, center_lon, 
        start_tile_x, start_tile_y, 
        zoom, tile_size
    )
    
    meters_per_pixel = 40075016.686 * math.cos(math.radians(center_lat)) / (256 * 2**zoom)
    radius_px = int(radius_meters / meters_per_pixel)
    
    # Draw radius circle
    draw.ellipse(
        [center_px - radius_px, center_py - radius_px,
         center_px + radius_px, center_py + radius_px],
        outline=(255, 255, 0, 200),
        width=3
    )
    
    # Draw building markers
    for lat, lon in building_centers:
        px, py = lat_lon_to_pixel(lat, lon, start_tile_x, start_tile_y, zoom, tile_size)
        draw.ellipse([px-3, py-3, px+3, py+3], fill=(255, 0, 0, 200))
    
    # Draw center marker
    draw.ellipse(
        [center_px - 8, center_py - 8, center_px + 8, center_py + 8],
        fill=(0, 255, 0, 255),
        outline=(0, 100, 0, 255)
    )
    
    # Add text
    font = _find_font(24)
    small_font = _find_font(16)
    
    overlay = Image.new("RGBA", composite.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    box_padding = 15
    box_x, box_y = 20, 20
    
    overlay_draw.rectangle([box_x, box_y, box_x + 320, box_y + 100], fill=(0, 0, 0, 180))
    overlay_draw.text((box_x + box_padding, box_y + box_padding), 
                      f"Buildings Found: {len(building_centers)}", fill=(255, 255, 255), font=font)
    overlay_draw.text((box_x + box_padding, box_y + box_padding + 30), 
                      f"Center: {center_lat:.6f}, {center_lon:.6f}", fill=(200, 200, 200), font=small_font)
    overlay_draw.text((box_x + box_padding, box_y + box_padding + 52), 
                      f"Radius: {radius_meters/1000:.1f} km | Zoom: {zoom}", fill=(200, 200, 200), font=small_font)
    
    composite = Image.alpha_composite(composite.convert("RGBA"), overlay)
    
    return composite.convert("RGB")
