"""
OpenStreetMap Overpass API module for querying buildings.
Used for comparison testing against Microsoft Building Footprints.
"""
import requests
from dataclasses import dataclass
from typing import List


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
    radius_meters: float
) -> List[dict]:
    """
    Query OSM for building polygons with geometry.
    Returns list of dicts compatible with the visualization module.
    """
    query = f"""
    [out:json][timeout:120];
    (
      way["building"~"^(house|residential|detached|semidetached_house|terrace|apartments|bungalow)$"](around:{radius_meters},{lat},{lon});
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
    
    return polygons
