"""
Microsoft Building Footprints module.
Queries building footprints from Overture Maps (which includes Microsoft's ML-derived buildings).
Uses the official overturemaps library for optimized access.
No API key required - data is publicly accessible.
"""
import math
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

import overturemaps


# ---------------------------------------------------------------------------
# Cache for Overture Maps query results (avoids repeated S3 round-trips)
# ---------------------------------------------------------------------------
_query_cache: Dict[tuple, Tuple[gpd.GeoDataFrame, float]] = {}
_CACHE_MAX_SIZE = 50
_CACHE_TTL_SECONDS = 300  # 5 minutes


@dataclass
class Building:
    """Represents a building from building footprints data."""
    id: int
    lat: float
    lon: float
    area_sqm: float
    geometry: any  # Shapely polygon


def get_bounding_box(lat: float, lon: float, radius_meters: float) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box from center point and radius.
    Returns (min_lon, min_lat, max_lon, max_lat).
    """
    R = 6371000  # Earth's radius in meters
    angular_distance = radius_meters / R

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    min_lat = math.degrees(lat_rad - angular_distance)
    max_lat = math.degrees(lat_rad + angular_distance)

    delta_lon = math.asin(math.sin(angular_distance) / math.cos(lat_rad))
    min_lon = math.degrees(lon_rad - delta_lon)
    max_lon = math.degrees(lon_rad + delta_lon)

    return (min_lon, min_lat, max_lon, max_lat)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_cache_key(lat: float, lon: float, radius_meters: float) -> tuple:
    """Deterministic cache key rounded to avoid floating-point duplicates."""
    return (round(lat, 6), round(lon, 6), round(radius_meters, 1))


def _get_utm_crs(lat: float, lon: float) -> str:
    """Return the UTM CRS string for the given lat/lon."""
    utm_zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + utm_zone}" if lat >= 0 else f"EPSG:{32700 + utm_zone}"


def _fetch_and_filter_buildings(
    lat: float, lon: float, radius_meters: float
) -> gpd.GeoDataFrame:
    """
    Core query: fetch buildings from Overture Maps, filter to circular radius.

    Results are cached for up to _CACHE_TTL_SECONDS so repeated / identical
    requests skip the S3 round-trip entirely.  A persistent disk cache
    (managed by CacheManager) is checked before hitting S3.
    """
    cache_key = _get_cache_key(lat, lon, radius_meters)

    # -- Check in-memory cache ------------------------------------------------
    if cache_key in _query_cache:
        cached_gdf, cached_time = _query_cache[cache_key]
        if time.time() - cached_time < _CACHE_TTL_SECONDS:
            print(f"Memory cache hit for ({lat}, {lon}, {radius_meters}m)")
            return cached_gdf
        del _query_cache[cache_key]

    # -- Check persistent disk cache ------------------------------------------
    bbox = get_bounding_box(lat, lon, radius_meters)
    try:
        from cache_manager import get_cache_manager
        disk_mgr = get_cache_manager()
        covering = disk_mgr.find_covering_cache(bbox)
        if covering:
            disk_gdf = disk_mgr.load_geodataframe(covering["id"])
            if disk_gdf is not None and len(disk_gdf) > 0:
                print(
                    f"Disk cache hit: '{covering['name']}' "
                    f"({covering['building_count']} buildings)"
                )
                # Spatial filter: bbox then circular radius
                min_lon, min_lat, max_lon, max_lat = bbox
                disk_gdf = disk_gdf.cx[min_lon:max_lon, min_lat:max_lat].copy()

                center = Point(lon, lat)
                utm_crs = _get_utm_crs(lat, lon)
                proj_utm = pyproj.Transformer.from_crs(
                    'EPSG:4326', utm_crs, always_xy=True
                ).transform
                proj_wgs = pyproj.Transformer.from_crs(
                    utm_crs, 'EPSG:4326', always_xy=True
                ).transform
                center_utm = transform(proj_utm, center)
                circle_utm = center_utm.buffer(radius_meters)
                circle_wgs = transform(proj_wgs, circle_utm)
                disk_gdf = disk_gdf[
                    disk_gdf.geometry.intersects(circle_wgs)
                ].copy()

                print(f"Disk cache: {len(disk_gdf)} buildings within radius")

                # Populate memory cache
                if len(_query_cache) >= _CACHE_MAX_SIZE:
                    oldest = min(_query_cache, key=lambda k: _query_cache[k][1])
                    del _query_cache[oldest]
                _query_cache[cache_key] = (disk_gdf, time.time())
                return disk_gdf
    except Exception as exc:
        print(f"Disk cache lookup failed (non-fatal): {exc}")

    # -- Query Overture Maps --------------------------------------------------
    min_lon, min_lat, max_lon, max_lat = bbox

    print(f"Querying Overture Maps for buildings in bbox: {bbox}")
    t0 = time.time()

    try:
        bbox_tuple = (min_lon, min_lat, max_lon, max_lat)
        gdf = overturemaps.record_batch_reader("building", bbox_tuple).read_all().to_pandas()
        print(f"Query returned {len(gdf)} buildings from Overture Maps ({time.time() - t0:.1f}s)")

        if len(gdf) == 0:
            empty = gpd.GeoDataFrame()
            _query_cache[cache_key] = (empty, time.time())
            return empty

        if 'geometry' in gdf.columns:
            from shapely import wkb
            if isinstance(gdf['geometry'].iloc[0], bytes):
                gdf['geometry'] = gdf['geometry'].apply(lambda x: wkb.loads(x) if x else None)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')
        else:
            print("No geometry column found")
            empty = gpd.GeoDataFrame()
            _query_cache[cache_key] = (empty, time.time())
            return empty

    except Exception as e:
        print(f"Error querying Overture Maps: {e}")
        import traceback
        traceback.print_exc()
        return gpd.GeoDataFrame()

    # -- Filter to circular radius -------------------------------------------
    center = Point(lon, lat)
    utm_crs = _get_utm_crs(lat, lon)

    project_to_utm = pyproj.Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True).transform
    project_to_wgs = pyproj.Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True).transform

    center_utm = transform(project_to_utm, center)
    search_circle_utm = center_utm.buffer(radius_meters)
    search_circle = transform(project_to_wgs, search_circle_utm)

    gdf = gdf[gdf.geometry.intersects(search_circle)].copy()
    print(f"Buildings within {radius_meters}m radius: {len(gdf)} ({time.time() - t0:.1f}s total)")

    # -- Cache the filtered result -------------------------------------------
    if len(_query_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_query_cache, key=lambda k: _query_cache[k][1])
        del _query_cache[oldest_key]
    _query_cache[cache_key] = (gdf, time.time())

    return gdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_buildings_in_radius(
    lat: float, lon: float, radius_meters: float
) -> Tuple[int, float, float]:
    """
    Fast count-only query — no Building objects, no per-row loops.

    Returns:
        (building_count, total_area_sqm, avg_area_sqm)
    """
    gdf = _fetch_and_filter_buildings(lat, lon, radius_meters)

    if len(gdf) == 0:
        return 0, 0.0, 0.0

    # Vectorized area calculation via bulk CRS projection
    utm_crs = _get_utm_crs(lat, lon)
    gdf_utm = gdf.to_crs(utm_crs)
    areas = gdf_utm.area

    count = len(gdf)
    total_area = float(areas.sum())
    avg_area = total_area / count

    return count, round(total_area, 2), round(avg_area, 2)


def query_ms_buildings_in_radius(
    lat: float,
    lon: float,
    radius_meters: float
) -> Tuple[List[Building], gpd.GeoDataFrame]:
    """
    Query building footprints and return Building objects + GeoDataFrame.

    Uses vectorized CRS projection and cached queries for speed.
    """
    gdf = _fetch_and_filter_buildings(lat, lon, radius_meters)

    if len(gdf) == 0:
        return [], gpd.GeoDataFrame()

    # Vectorized projection & area calculation (one bulk operation, not per-row)
    utm_crs = _get_utm_crs(lat, lon)
    gdf_utm = gdf.to_crs(utm_crs)
    area_values = gdf_utm.area.values
    centroids = gdf.geometry.centroid

    buildings = []
    for i, (idx, row) in enumerate(gdf.iterrows()):
        try:
            buildings.append(Building(
                id=hash(str(row.get('id', idx))) % (10**9),
                lat=centroids.iloc[i].y,
                lon=centroids.iloc[i].x,
                area_sqm=float(area_values[i]),
                geometry=row.geometry
            ))
        except Exception:
            continue

    return buildings, gdf


def get_building_polygons_ms(
    lat: float,
    lon: float,
    radius_meters: float
) -> List[dict]:
    """
    Get building polygons for visualization.
    Returns list of dicts compatible with the visualization module.
    """
    buildings, gdf = query_ms_buildings_in_radius(lat, lon, radius_meters)

    polygons = []
    for building in buildings:
        try:
            if building.geometry.geom_type == 'Polygon':
                coords = [(y, x) for x, y in building.geometry.exterior.coords]
            elif building.geometry.geom_type == 'MultiPolygon':
                largest = max(building.geometry.geoms, key=lambda g: g.area)
                coords = [(y, x) for x, y in largest.exterior.coords]
            else:
                continue
        except Exception:
            continue

        polygons.append({
            "id": building.id,
            "coordinates": coords,
            "type": "building",
            "center": (building.lat, building.lon),
            "area_sqm": building.area_sqm
        })

    return polygons
