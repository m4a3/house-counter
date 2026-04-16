"""
OpenStreetMap data fetcher module

Fetches building footprints and road networks from OpenStreetMap
using the Overpass API and OSMnx library.

Includes caching support to avoid re-fetching OSM data for the same area.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, box, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union
import warnings

import config

# Suppress some OSMnx warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure OSMnx — cache Overpass HTTP responses so repeated queries are instant
ox.settings.use_cache = True
ox.settings.cache_folder = str(Path(__file__).parent.parent / "output" / "cache" / "osmnx_http")
ox.settings.log_console = False
ox.settings.timeout = 60

# Cache directory
OSM_CACHE_DIR = Path(config.OUTPUT_DIR) / "cache" / "osm"


def get_osm_cache_key(center_lat: float, center_lon: float, radius_m: float) -> str:
    """Generate a unique cache key for OSM data."""
    key_data = f"{center_lat:.6f}_{center_lon:.6f}_{radius_m:.1f}"
    return hashlib.md5(key_data.encode()).hexdigest()[:12]


def load_osm_from_cache(center_lat: float, center_lon: float, radius_m: float) -> Optional[Dict[str, gpd.GeoDataFrame]]:
    """Try to load OSM data from cache."""
    cache_key = get_osm_cache_key(center_lat, center_lon, radius_m)
    buildings_path = OSM_CACHE_DIR / f"{cache_key}_buildings.geojson"
    roads_path = OSM_CACHE_DIR / f"{cache_key}_roads.geojson"
    driveways_path = OSM_CACHE_DIR / f"{cache_key}_driveways.geojson"
    exclusions_path = OSM_CACHE_DIR / f"{cache_key}_exclusions.geojson"
    boundaries_path = OSM_CACHE_DIR / f"{cache_key}_boundaries.geojson"
    
    addresses_path = OSM_CACHE_DIR / f"{cache_key}_addresses.geojson"
    
    # Need all files for complete cache hit (including address polygons)
    if buildings_path.exists() and roads_path.exists() and driveways_path.exists() and boundaries_path.exists() and addresses_path.exists():
        try:
            buildings = gpd.read_file(buildings_path)
            roads = gpd.read_file(roads_path)
            driveways = gpd.read_file(driveways_path)
            exclusion_zones = gpd.read_file(exclusions_path) if exclusions_path.exists() else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            property_boundaries = gpd.read_file(boundaries_path)
            address_polygons = gpd.read_file(addresses_path)
            return {
                "buildings": buildings,
                "roads": roads,
                "driveways": driveways,
                "exclusion_zones": exclusion_zones,
                "property_boundaries": property_boundaries,
                "address_polygons": address_polygons
            }
        except Exception:
            return None
    return None


def save_osm_to_cache(data: Dict[str, gpd.GeoDataFrame], center_lat: float, center_lon: float, radius_m: float):
    """Save OSM data to cache."""
    OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_osm_cache_key(center_lat, center_lon, radius_m)
    
    def save_gdf(gdf, path):
        if gdf is not None and not gdf.empty:
            gdf.to_file(path, driver="GeoJSON")
        else:
            gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(path, driver="GeoJSON")
    
    try:
        save_gdf(data.get("buildings"), OSM_CACHE_DIR / f"{cache_key}_buildings.geojson")
        save_gdf(data.get("roads"), OSM_CACHE_DIR / f"{cache_key}_roads.geojson")
        save_gdf(data.get("driveways"), OSM_CACHE_DIR / f"{cache_key}_driveways.geojson")
        save_gdf(data.get("exclusion_zones"), OSM_CACHE_DIR / f"{cache_key}_exclusions.geojson")
        save_gdf(data.get("property_boundaries"), OSM_CACHE_DIR / f"{cache_key}_boundaries.geojson")
        save_gdf(data.get("address_polygons"), OSM_CACHE_DIR / f"{cache_key}_addresses.geojson")
    except Exception:
        pass  # Silently fail on cache save errors


def fetch_buildings(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    show_progress: bool = True,
    exclude_outbuildings: bool = True
) -> gpd.GeoDataFrame:
    """
    Fetch building footprints from OpenStreetMap.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        show_progress: Show progress indicator
        exclude_outbuildings: If True, exclude sheds, garages, and other outbuildings
        
    Returns:
        GeoDataFrame with building polygons
    """
    from tqdm import tqdm
    
    if show_progress:
        pbar = tqdm(total=100, desc="Fetching buildings", unit="%")
        pbar.update(10)
    
    try:
        # Create a point and buffer for the search area
        center_point = (center_lat, center_lon)
        
        if show_progress:
            pbar.update(20)
        
        # Fetch buildings using OSMnx
        buildings = ox.features_from_point(
            center_point,
            tags={"building": True},
            dist=radius_m
        )
        
        if show_progress:
            pbar.update(40)
        
        # Filter to only polygons (some might be points or other geometries)
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
        
        # Keep useful columns
        cols_to_keep = ["geometry"]
        for col in ["building", "name", "addr:housenumber", "addr:street"]:
            if col in buildings.columns:
                cols_to_keep.append(col)
        
        buildings = buildings[cols_to_keep].reset_index(drop=True)
        
        # Filter out outbuildings (sheds, garages, etc.) - these are in gardens!
        if exclude_outbuildings and "building" in buildings.columns:
            outbuilding_types = [
                "shed", "garage", "garages", "carport", "outbuilding",
                "barn", "greenhouse", "hut", "cabin", "storage",
                "roof", "canopy", "shelter", "kiosk", "toilets",
                "transformer_tower", "service", "ruins"
            ]
            original_count = len(buildings)
            buildings = buildings[~buildings["building"].isin(outbuilding_types)].reset_index(drop=True)
            
            # Also filter by size - very small buildings are likely sheds even if not tagged
            # A typical shed is < 20 sq meters, a house is > 50 sq meters
            if not buildings.empty:
                # Project to meters for accurate area calculation
                buildings_m = buildings.to_crs(epsg=32629)  # UTM zone 29N for Ireland
                buildings["area_m2"] = buildings_m.geometry.area
                
                # Keep buildings > 30 sq meters (filters out small sheds tagged as "yes")
                size_filtered = buildings[buildings["area_m2"] > 30].reset_index(drop=True)
                
                if len(size_filtered) > 0:
                    excluded_by_size = len(buildings) - len(size_filtered)
                    buildings = size_filtered
                    if show_progress and excluded_by_size > 0:
                        print(f"    Also excluded {excluded_by_size} small structures (<30m²)")
            
            if show_progress:
                print(f"    Filtered to {len(buildings)} main buildings (excluded {original_count - len(buildings)} outbuildings)")
        
        if show_progress:
            pbar.update(30)
            pbar.close()
            if not exclude_outbuildings:
                print(f"    Found {len(buildings)} buildings")
        
        return buildings
        
    except Exception as e:
        if show_progress:
            pbar.close()
        print(f"Error fetching buildings: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_roads(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    show_progress: bool = True,
    front_facing_only: bool = True
) -> gpd.GeoDataFrame:
    """
    Fetch road network from OpenStreetMap.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        show_progress: Show progress indicator
        front_facing_only: If True, only return roads that determine "front" 
                          (residential, tertiary, secondary, primary roads)
        
    Returns:
        GeoDataFrame with road LineStrings
    """
    from tqdm import tqdm
    
    if show_progress:
        pbar = tqdm(total=100, desc="Fetching roads", unit="%")
        pbar.update(10)
    
    try:
        center_point = (center_lat, center_lon)
        
        if show_progress:
            pbar.update(20)
        
        # Fetch street network
        # network_type="all" includes all roads, not just driveable
        G = ox.graph_from_point(
            center_point,
            dist=radius_m,
            network_type="all",
            simplify=True
        )
        
        if show_progress:
            pbar.update(40)
        
        # Convert to GeoDataFrame of edges (roads)
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        
        # Keep useful columns
        cols_to_keep = ["geometry"]
        for col in ["name", "highway", "lanes", "oneway"]:
            if col in edges.columns:
                cols_to_keep.append(col)
        
        roads = edges[cols_to_keep].reset_index(drop=True)
        
        # Filter to only front-facing roads (exclude footways, service roads, etc.)
        # OSM can return highway as a list when a road has multiple tags
        # (e.g. ['residential', 'footway']).  We keep the road if ANY tag
        # matches a front-facing type.
        if front_facing_only and "highway" in roads.columns:
            front_road_types = {
                "residential", "tertiary", "secondary", "primary",
                "tertiary_link", "secondary_link", "primary_link",
                "living_street", "unclassified"
            }
            original_count = len(roads)
            
            def _is_front_road(hw):
                if isinstance(hw, list):
                    return any(t in front_road_types for t in hw)
                return hw in front_road_types
            
            roads = roads[roads["highway"].apply(_is_front_road)].reset_index(drop=True)
            if show_progress:
                print(f"    Filtered to {len(roads)} front-facing roads (from {original_count} total)")
        
        if show_progress:
            pbar.update(30)
            pbar.close()
            if not front_facing_only:
                print(f"    Found {len(roads)} road segments")
        
        return roads
        
    except Exception as e:
        if show_progress:
            pbar.close()
        print(f"Error fetching roads: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_exclusion_zones(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    show_progress: bool = True
) -> gpd.GeoDataFrame:
    """
    Fetch areas that should be excluded from garden classification.
    
    This includes parks, sports pitches, public grass areas, etc.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        show_progress: Show progress indicator
        
    Returns:
        GeoDataFrame with exclusion zone polygons
    """
    from tqdm import tqdm
    
    if show_progress:
        pbar = tqdm(total=100, desc="Fetching exclusion zones", unit="%")
        pbar.update(10)
    
    center_point = (center_lat, center_lon)
    exclusion_gdf_list = []
    
    try:
        # Fetch landuse areas (parks, grass, recreation, commercial, etc.)
        if show_progress:
            pbar.update(20)
        
        try:
            landuse = ox.features_from_point(
                center_point,
                tags={"landuse": [
                    "grass", "recreation_ground", "meadow", "village_green",
                    "allotments", "cemetery", "forest", "farmland", "farmyard",
                    "industrial", "commercial", "retail", "construction"
                ]},
                dist=radius_m
            )
            landuse = landuse[landuse.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not landuse.empty:
                landuse = landuse[["geometry"]].copy()
                landuse["exclusion_type"] = "landuse"
                exclusion_gdf_list.append(landuse)
        except Exception:
            pass
        
        if show_progress:
            pbar.update(20)
        
        # Fetch leisure areas (pitches, playgrounds, parks, sports)
        try:
            leisure = ox.features_from_point(
                center_point,
                tags={"leisure": [
                    "pitch", "playground", "park", "sports_centre", "track",
                    "golf_course", "nature_reserve", "common", "dog_park",
                    "fitness_station", "horse_riding"
                ]},
                dist=radius_m
            )
            leisure = leisure[leisure.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not leisure.empty:
                leisure = leisure[["geometry"]].copy()
                leisure["exclusion_type"] = "leisure"
                exclusion_gdf_list.append(leisure)
        except Exception:
            pass
        
        if show_progress:
            pbar.update(10)
        
        # Fetch natural areas (woods, water, scrub)
        try:
            natural = ox.features_from_point(
                center_point,
                tags={"natural": ["wood", "water", "scrub", "wetland", "heath"]},
                dist=radius_m
            )
            natural = natural[natural.geometry.type.isin(["Polygon", "MultiPolygon"])]
            if not natural.empty:
                natural = natural[["geometry"]].copy()
                natural["exclusion_type"] = "natural"
                exclusion_gdf_list.append(natural)
        except Exception:
            pass
        
        if show_progress:
            pbar.update(30)
        
        # Combine all exclusion zones
        if exclusion_gdf_list:
            exclusions = gpd.GeoDataFrame(
                pd.concat(exclusion_gdf_list, ignore_index=True),
                crs="EPSG:4326"
            )
        else:
            exclusions = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        if show_progress:
            pbar.update(10)
            pbar.close()
            print(f"    Found {len(exclusions)} exclusion zones (parks, pitches, etc.)")
        
        return exclusions
        
    except Exception as e:
        if show_progress:
            pbar.close()
        print(f"Error fetching exclusion zones: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_address_polygons(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    show_progress: bool = True
) -> gpd.GeoDataFrame:
    """
    Fetch address polygons from OpenStreetMap.
    
    These are building footprints with address information.
    The street name tells us which direction the building faces.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        show_progress: Show progress indicator
        
    Returns:
        GeoDataFrame with address polygons and street info
    """
    from tqdm import tqdm
    
    if show_progress:
        pbar = tqdm(total=100, desc="Fetching address data", unit="%")
        pbar.update(10)
    
    try:
        center_point = (center_lat, center_lon)
        
        if show_progress:
            pbar.update(30)
        
        # Fetch features with address information
        addresses = ox.features_from_point(
            center_point,
            tags={"addr:housenumber": True},
            dist=radius_m
        )
        
        if show_progress:
            pbar.update(40)
        
        # Filter to polygons only (building footprints with addresses)
        addresses = addresses[addresses.geometry.type.isin(["Polygon", "MultiPolygon"])]
        
        # Keep useful columns
        cols_to_keep = ["geometry"]
        for col in ["addr:housenumber", "addr:street", "addr:city"]:
            if col in addresses.columns:
                cols_to_keep.append(col)
        
        addresses = addresses[cols_to_keep].reset_index(drop=True)
        
        if show_progress:
            pbar.update(20)
            pbar.close()
            print(f"    Found {len(addresses)} address polygons")
        
        return addresses
        
    except Exception as e:
        if show_progress:
            pbar.close()
        print(f"Error fetching address data: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_property_boundaries(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    show_progress: bool = True
) -> gpd.GeoDataFrame:
    """
    Fetch property boundaries (walls, fences, hedges) from OpenStreetMap.
    
    These define the edges between properties and are crucial for
    separating front/back gardens between different houses.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        show_progress: Show progress indicator
        
    Returns:
        GeoDataFrame with boundary LineStrings/Polygons
    """
    from tqdm import tqdm
    
    if show_progress:
        pbar = tqdm(total=100, desc="Fetching property boundaries", unit="%")
        pbar.update(10)
    
    try:
        center_point = (center_lat, center_lon)
        
        if show_progress:
            pbar.update(30)
        
        # Fetch barriers (walls, fences, hedges)
        barriers = ox.features_from_point(
            center_point,
            tags={"barrier": ["fence", "wall", "hedge", "retaining_wall"]},
            dist=radius_m
        )
        
        if show_progress:
            pbar.update(40)
        
        # Keep all geometry types (lines and polygons)
        if not barriers.empty:
            barriers = barriers[["geometry"]].reset_index(drop=True)
        else:
            barriers = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        if show_progress:
            pbar.update(20)
            pbar.close()
            print(f"    Found {len(barriers)} property boundaries (walls/fences)")
        
        return barriers
        
    except Exception as e:
        if show_progress:
            pbar.close()
        print(f"Error fetching property boundaries: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_driveways(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    show_progress: bool = True
) -> gpd.GeoDataFrame:
    """
    Fetch driveways from OpenStreetMap.
    
    Driveways are key for determining the front of a building!
    They connect the road to the front of the house.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        show_progress: Show progress indicator
        
    Returns:
        GeoDataFrame with driveway LineStrings
    """
    from tqdm import tqdm
    
    if show_progress:
        pbar = tqdm(total=100, desc="Fetching driveways", unit="%")
        pbar.update(10)
    
    try:
        center_point = (center_lat, center_lon)
        
        if show_progress:
            pbar.update(30)
        
        # Fetch service roads with driveway tag
        driveways = ox.features_from_point(
            center_point,
            tags={"highway": "service", "service": "driveway"},
            dist=radius_m
        )
        
        if show_progress:
            pbar.update(40)
        
        # Filter to LineStrings
        driveways = driveways[driveways.geometry.type.isin(["LineString", "MultiLineString"])]
        
        # Keep geometry only
        driveways = driveways[["geometry"]].reset_index(drop=True)
        
        if show_progress:
            pbar.update(20)
            pbar.close()
            print(f"    Found {len(driveways)} driveways")
        
        return driveways
        
    except Exception as e:
        if show_progress:
            pbar.close()
        print(f"Error fetching driveways: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_osm_features(
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Single Overpass API request for ALL OSM data needed.

    Tags are merged by OSMnx into one QL query — one HTTP round-trip.
    Roads (highway ways) are parsed from the same response; there is no
    second graph_from_point call.  This avoids Overpass rate-limiting
    caused by two simultaneous requests from the same IP.
    """
    _empty = lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    ox.settings.timeout = 60

    # Only fetch the highway types we actually use for front/back classification.
    # "highway": True would fetch every traffic-light node, bus stop, etc. —
    # thousands of unwanted point features that bloat the response massively.
    _road_types = [
        "residential", "tertiary", "secondary", "primary",
        "tertiary_link", "secondary_link", "primary_link",
        "living_street", "unclassified",
        "service",  # includes driveways (service + service=driveway)
    ]

    try:
        all_feats = ox.features_from_point(
            (center_lat, center_lon),
            tags={
                "building": True,
                "highway": _road_types,
                "addr:housenumber": True,
                "barrier": ["fence", "wall", "hedge", "retaining_wall"],
            },
            dist=radius_m,
        )
    except Exception:
        all_feats = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _col(df, name):
        return df[name] if name in df.columns else pd.Series(dtype=object, index=df.index)

    # ── Buildings ────────────────────────────────────────────────────────────
    try:
        bld_mask = _col(all_feats, "building").notna()
        buildings = all_feats[bld_mask & all_feats.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        cols = ["geometry"] + [c for c in ["building", "name", "addr:housenumber", "addr:street"] if c in buildings.columns]
        buildings = buildings[cols].reset_index(drop=True)

        if "building" in buildings.columns:
            outbuilding_types = {
                "shed", "garage", "garages", "carport", "outbuilding",
                "barn", "greenhouse", "hut", "cabin", "storage",
                "roof", "canopy", "shelter", "kiosk", "toilets",
                "transformer_tower", "service", "ruins",
            }
            buildings = buildings[~buildings["building"].isin(outbuilding_types)].reset_index(drop=True)

        if not buildings.empty:
            bld_m = buildings.to_crs(epsg=32629)
            buildings = buildings[bld_m.geometry.area > 30].reset_index(drop=True)
    except Exception:
        buildings = _empty()

    # ── Roads — parsed from the same response, no second Overpass call ────────
    front_road_types = {
        "residential", "tertiary", "secondary", "primary",
        "tertiary_link", "secondary_link", "primary_link",
        "living_street", "unclassified",
    }
    try:
        hw_col = _col(all_feats, "highway")
        road_mask = hw_col.notna() & all_feats.geometry.type.isin(["LineString", "MultiLineString"])
        roads_raw = all_feats[road_mask].copy()
        cols = ["geometry"] + [c for c in ["highway", "name", "lanes", "oneway"] if c in roads_raw.columns]
        roads_raw = roads_raw[cols].reset_index(drop=True)

        if "highway" in roads_raw.columns:
            def _is_front(hw):
                if isinstance(hw, list):
                    return any(t in front_road_types for t in hw)
                return hw in front_road_types
            roads = roads_raw[roads_raw["highway"].apply(_is_front)].reset_index(drop=True)
        else:
            roads = roads_raw
    except Exception:
        roads = _empty()

    # ── Driveways ────────────────────────────────────────────────────────────
    try:
        hw_col  = _col(all_feats, "highway")
        svc_col = _col(all_feats, "service")
        driveways = all_feats[
            (hw_col == "service") &
            (svc_col == "driveway") &
            all_feats.geometry.type.isin(["LineString", "MultiLineString"])
        ][["geometry"]].reset_index(drop=True)
    except Exception:
        driveways = _empty()

    # ── Address polygons ──────────────────────────────────────────────────────
    try:
        addr_mask = _col(all_feats, "addr:housenumber").notna()
        address_polygons = all_feats[addr_mask & all_feats.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        cols = ["geometry"] + [c for c in ["addr:housenumber", "addr:street", "addr:city"] if c in address_polygons.columns]
        address_polygons = address_polygons[cols].reset_index(drop=True)
    except Exception:
        address_polygons = _empty()

    # ── Property boundaries ───────────────────────────────────────────────────
    try:
        barrier_col = _col(all_feats, "barrier")
        property_boundaries = all_feats[barrier_col.notna()][["geometry"]].reset_index(drop=True)
    except Exception:
        property_boundaries = _empty()

    return {
        "buildings": buildings,
        "roads": roads,
        "driveways": driveways,
        "address_polygons": address_polygons,
        "property_boundaries": property_boundaries,
    }


def fetch_osm_batch(
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> Dict[str, gpd.GeoDataFrame]:
    """Convenience wrapper — delegates to fetch_osm_features (single request)."""
    return fetch_osm_features(center_lat, center_lon, radius_m)


def fetch_all_osm_data(
    center_lat: float,
    center_lon: float,
    radius_m: float = 500,
    use_cache: bool = True
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Fetch all required OSM data for garden classification.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Radius in meters
        use_cache: Whether to use cached data if available
        
    Returns:
        Dict with 'buildings', 'roads', 'driveways', 'exclusion_zones', 'property_boundaries' GeoDataFrames
    """
    # Try to load from cache first
    if use_cache:
        cached = load_osm_from_cache(center_lat, center_lon, radius_m)
        if cached is not None and "address_polygons" in cached:
            print(f"📦 Loaded OSM data from cache ({len(cached['buildings'])} buildings, {len(cached['roads'])} roads, {len(cached['driveways'])} driveways, {len(cached['address_polygons'])} addresses)")
            return cached
    
    # Fetch all data types
    buildings = fetch_buildings(center_lat, center_lon, radius_m)
    roads = fetch_roads(center_lat, center_lon, radius_m)
    driveways = fetch_driveways(center_lat, center_lon, radius_m)
    exclusion_zones = fetch_exclusion_zones(center_lat, center_lon, radius_m)
    property_boundaries = fetch_property_boundaries(center_lat, center_lon, radius_m)
    address_polygons = fetch_address_polygons(center_lat, center_lon, radius_m)
    
    data = {
        "buildings": buildings,
        "roads": roads,
        "driveways": driveways,
        "exclusion_zones": exclusion_zones,
        "property_boundaries": property_boundaries,
        "address_polygons": address_polygons,
    }
    
    # Save to cache
    if use_cache and len(buildings) > 0:
        save_osm_to_cache(data, center_lat, center_lon, radius_m)
        print(f"💾 Saved OSM data to cache")
    
    return data


def project_to_meters(gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float) -> gpd.GeoDataFrame:
    """
    Project GeoDataFrame to a local meter-based CRS for accurate distance calculations.
    
    Uses UTM zone appropriate for the location.
    
    Args:
        gdf: GeoDataFrame in EPSG:4326
        center_lat: Center latitude (for UTM zone selection)
        center_lon: Center longitude (for UTM zone selection)
        
    Returns:
        GeoDataFrame in meter-based CRS
    """
    if gdf.empty:
        return gdf
        
    # Determine UTM zone
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    
    return gdf.to_crs(epsg=epsg)


def create_road_buffer(roads: gpd.GeoDataFrame, buffer_m: float = 5.0) -> gpd.GeoDataFrame:
    """
    Create buffered polygons around roads for easier spatial operations.
    
    Args:
        roads: GeoDataFrame with road LineStrings (in meter-based CRS)
        buffer_m: Buffer distance in meters
        
    Returns:
        GeoDataFrame with road buffer polygons
    """
    if roads.empty:
        return gpd.GeoDataFrame(geometry=[], crs=roads.crs)
    
    # Buffer each road
    buffered = roads.copy()
    buffered["geometry"] = roads.geometry.buffer(buffer_m)
    
    return buffered


def get_building_centroids(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Get centroids of buildings for distance calculations.
    
    Args:
        buildings: GeoDataFrame with building polygons
        
    Returns:
        GeoDataFrame with building centroid points
    """
    if buildings.empty:
        return gpd.GeoDataFrame(geometry=[], crs=buildings.crs)
    
    centroids = buildings.copy()
    centroids["geometry"] = buildings.geometry.centroid
    
    return centroids


def find_nearest_road_direction(
    building: Polygon,
    roads: gpd.GeoDataFrame
) -> Optional[np.ndarray]:
    """
    Find the direction vector from a building to its nearest road.
    
    Args:
        building: Building polygon
        roads: GeoDataFrame with road geometries
        
    Returns:
        Unit vector pointing from building centroid to nearest road point,
        or None if no roads found
    """
    if roads.empty:
        return None
    
    centroid = building.centroid
    
    # Find nearest point on any road
    min_dist = float("inf")
    nearest_point = None
    
    for road_geom in roads.geometry:
        if road_geom.is_empty:
            continue
        nearest = road_geom.interpolate(road_geom.project(centroid))
        dist = centroid.distance(nearest)
        if dist < min_dist:
            min_dist = dist
            nearest_point = nearest
    
    if nearest_point is None:
        return None
    
    # Calculate direction vector
    dx = nearest_point.x - centroid.x
    dy = nearest_point.y - centroid.y
    
    # Normalize
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-6:
        return None
    
    return np.array([dx / length, dy / length])


def geo_to_pixel(
    lat: float,
    lon: float,
    geo_bounds: dict,
    image_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        geo_bounds: Dict with 'north', 'south', 'east', 'west'
        image_size: Tuple of (width, height)
        
    Returns:
        Tuple of (pixel_x, pixel_y)
    """
    width, height = image_size
    
    # Normalize to 0-1 range
    x_norm = (lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"])
    y_norm = (geo_bounds["north"] - lat) / (geo_bounds["north"] - geo_bounds["south"])
    
    # Convert to pixels
    px = int(x_norm * width)
    py = int(y_norm * height)
    
    return px, py


def geometry_to_pixel_coords(
    geometry,
    geo_bounds: dict,
    image_size: Tuple[int, int]
) -> list:
    """
    Convert a Shapely geometry to pixel coordinates.
    
    Args:
        geometry: Shapely geometry (Polygon, LineString, etc.)
        geo_bounds: Dict with 'north', 'south', 'east', 'west'
        image_size: Tuple of (width, height)
        
    Returns:
        List of pixel coordinate tuples
    """
    if geometry.is_empty:
        return []
    
    if isinstance(geometry, (Polygon, MultiPolygon)):
        if isinstance(geometry, MultiPolygon):
            # Use the largest polygon
            geometry = max(geometry.geoms, key=lambda g: g.area)
        coords = list(geometry.exterior.coords)
    elif isinstance(geometry, LineString):
        coords = list(geometry.coords)
    else:
        return []
    
    pixel_coords = []
    for lon, lat in coords:
        px, py = geo_to_pixel(lat, lon, geo_bounds, image_size)
        pixel_coords.append((px, py))
    
    return pixel_coords
