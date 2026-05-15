"""
OpenStreetMap Overpass API module for querying buildings.
Used for comparison testing against Microsoft Building Footprints.
"""
import time
import requests
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class OSMBuilding:
    """Represents a building from OSM data."""
    id: int
    lat: float
    lon: float
    building_type: str
    name: str | None = None


# Multiple Overpass endpoints for fallback
OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


# ---------------------------------------------------------------------------
# In-memory cache for OSM polygon queries.
#
# Overpass is flaky and rate-limited, especially for large queries. Caching
# repeats of the same (lat, lon, radius, all_buildings) tuple lets the
# contributor UI re-detect in the same area without hammering Overpass —
# and survives transient Overpass outages within the TTL window.
# ---------------------------------------------------------------------------
_osm_cache: Dict[tuple, Tuple[List[dict], float]] = {}
_OSM_CACHE_MAX_SIZE = 50
_OSM_CACHE_TTL_SECONDS = 300  # 5 minutes


def _osm_cache_key(lat: float, lon: float, radius_meters: float, all_buildings: bool) -> tuple:
    return (round(lat, 6), round(lon, 6), round(radius_meters, 1), bool(all_buildings))


def _osm_cache_get(key: tuple) -> List[dict] | None:
    entry = _osm_cache.get(key)
    if not entry:
        return None
    polygons, ts = entry
    if time.time() - ts > _OSM_CACHE_TTL_SECONDS:
        del _osm_cache[key]
        return None
    return polygons


def _osm_cache_put(key: tuple, polygons: List[dict]) -> None:
    if len(_osm_cache) >= _OSM_CACHE_MAX_SIZE:
        oldest = min(_osm_cache, key=lambda k: _osm_cache[k][1])
        del _osm_cache[oldest]
    _osm_cache[key] = (polygons, time.time())


def _query_overpass(query: str) -> dict:
    """Try multiple Overpass endpoints with fallback."""
    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = requests.post(
                endpoint, 
                data={"data": query}, 
                timeout=120,
                headers={"User-Agent": "HouseCounter/1.0"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            last_error = e
            continue
    raise last_error or Exception("All Overpass endpoints failed")


def query_osm_buildings(
    lat: float, 
    lon: float, 
    radius_meters: float
) -> List[OSMBuilding]:
    """
    Query OSM for residential buildings within a radius.
    
    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius_meters: Search radius in meters
        
    Returns:
        List of OSMBuilding objects found within the radius
    """
    # Query for residential building types
    query = f"""
    [out:json][timeout:120];
    (
      way["building"~"^(house|residential|detached|semidetached_house|terrace|apartments|bungalow)$"](around:{radius_meters},{lat},{lon});
    );
    out center;
    """
    
    data = _query_overpass(query)
    
    buildings = []
    seen_ids = set()
    
    for element in data.get("elements", []):
        element_id = element.get("id")
        if element_id in seen_ids:
            continue
        seen_ids.add(element_id)
        
        # Get coordinates - for ways, use center
        if element.get("type") == "way":
            center = element.get("center", {})
            elem_lat = center.get("lat")
            elem_lon = center.get("lon")
        else:
            elem_lat = element.get("lat")
            elem_lon = element.get("lon")
            
        if elem_lat is None or elem_lon is None:
            continue
            
        tags = element.get("tags", {})
        building_type = tags.get("building", "unknown")
        name = tags.get("name") or tags.get("addr:street")
        
        buildings.append(OSMBuilding(
            id=element_id,
            lat=elem_lat,
            lon=elem_lon,
            building_type=building_type,
            name=name
        ))
    
    return buildings


def get_osm_building_polygons(
    lat: float,
    lon: float,
    radius_meters: float,
    all_buildings: bool = False,
    bypass_cache: bool = False,
) -> List[dict]:
    """
    Query OSM for building polygons with geometry.
    Returns list of dicts compatible with the visualization module.

    Args:
        lat: Latitude of center point
        lon: Longitude of center point
        radius_meters: Search radius in meters
        all_buildings: When True, return every OSM ``building=*`` way (any
            value). When False (default), restrict to residential building
            types — matches the behaviour of ``query_osm_buildings`` so the
            ``/compare`` counts stay apples-to-apples.
        bypass_cache: When True, skip the in-memory cache and always hit
            Overpass. When False (default), repeat queries within the TTL
            window are served from memory.
    """
    cache_key = _osm_cache_key(lat, lon, radius_meters, all_buildings)
    if not bypass_cache:
        cached = _osm_cache_get(cache_key)
        if cached is not None:
            print(f"OSM cache hit ({len(cached)} polygons) for {cache_key}")
            return cached

    building_filter = (
        '["building"]'
        if all_buildings
        else '["building"~"^(house|residential|detached|semidetached_house|terrace|apartments|bungalow)$"]'
    )
    query = f"""
    [out:json][timeout:120];
    (
      way{building_filter}(around:{radius_meters},{lat},{lon});
    );
    out body geom;
    """
    
    data = _query_overpass(query)
    
    polygons = []
    
    for element in data.get("elements", []):
        if element.get("type") != "way":
            continue
            
        geometry = element.get("geometry", [])
        if not geometry:
            continue
            
        coords = [(pt["lat"], pt["lon"]) for pt in geometry]
        tags = element.get("tags", {})
        
        polygons.append({
            "id": element.get("id"),
            "coordinates": coords,
            "type": tags.get("building", "unknown"),
            "center": (
                sum(c[0] for c in coords) / len(coords),
                sum(c[1] for c in coords) / len(coords)
            )
        })
    
    _osm_cache_put(cache_key, polygons)
    return polygons


def osm_cache_stats() -> dict:
    """Return summary of the in-memory OSM polygon cache."""
    now = time.time()
    entries = []
    for key, (polygons, ts) in _osm_cache.items():
        age = now - ts
        ttl_remaining = _OSM_CACHE_TTL_SECONDS - age
        entries.append({
            "lat": key[0], "lon": key[1], "radius_meters": key[2],
            "all_buildings": key[3], "polygon_count": len(polygons),
            "age_seconds": round(age, 1),
            "ttl_remaining_seconds": round(max(0.0, ttl_remaining), 1),
        })
    return {
        "size": len(_osm_cache),
        "max_size": _OSM_CACHE_MAX_SIZE,
        "ttl_seconds": _OSM_CACHE_TTL_SECONDS,
        "entries": entries,
    }


def osm_cache_clear() -> int:
    """Drop every cached OSM polygon set. Returns the number cleared."""
    n = len(_osm_cache)
    _osm_cache.clear()
    return n
