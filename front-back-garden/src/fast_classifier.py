"""
Fast Garden Classifier Module

Optimized for quick single-point and small-area queries.
Uses caching and reduced processing for speed.

Key optimizations:
- Smaller processing radius for point queries
- Skip visualization steps
- Cache at coordinate level, not chunk level
- Async-friendly design
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import config
from src.tiles import TileSource, fetch_area_image
from src.osm import (
    fetch_all_osm_data, geometry_to_pixel_coords, project_to_meters, 
    OSM_CACHE_DIR, get_osm_cache_key, fetch_buildings, fetch_roads, fetch_driveways,
    fetch_address_polygons, fetch_property_boundaries
)
from src.garden_detector import detect_green_areas, exclude_buildings_from_mask, exclude_roads_from_mask, split_vegetation_by_texture, recover_shaded_grass
from src.classifier import GardenClassifier
from src.delivery_pins import DeliveryPinFinder, SurfaceType
from src.precompute import PrecomputeManager

# Cache directory for fast lookups
FAST_CACHE_DIR = Path(config.OUTPUT_DIR) / "fast_cache"


@dataclass
class CachedResult:
    """Cached result for a specific coordinate."""
    lat: float
    lon: float
    front_pin: Optional[Dict[str, Any]]
    back_pin: Optional[Dict[str, Any]]
    classification: str
    surface_type: str
    score: float
    distance_to_building_m: float
    computed_at: float
    processing_time: float


class FastGardenClassifier:
    """
    Optimized classifier for quick API responses.
    
    Key differences from PrecomputeManager:
    - Uses smaller radius (50m) for single-point queries
    - Caches by precise coordinate hash, not chunk grid
    - Skips unnecessary processing steps
    - Designed for concurrent access
    """
    
    # Small radius for point queries - just enough to find nearest building
    POINT_QUERY_RADIUS = 50  # meters
    
    # Cache TTL (time to live) in seconds - 24 hours
    CACHE_TTL = 86400
    
    def __init__(
        self,
        tile_source: TileSource = TileSource.AUTO,
        zoom: int = 19,
        cache_dir: Path = None
    ):
        self.tile_source = tile_source
        self.zoom = zoom
        self.cache_dir = cache_dir or FAST_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, lat: float, lon: float) -> str:
        """Generate precise cache key for coordinates."""
        # Use 5 decimal places (~1m precision)
        key_data = f"{lat:.5f}_{lon:.5f}_{self.zoom}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, lat: float, lon: float) -> Optional[CachedResult]:
        """Load result from cache if valid."""
        cache_key = self._get_cache_key(lat, lon)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                
                # CRITICAL: Validate cached coordinates match request
                # This prevents returning wrong results from hash collisions or old cache
                if abs(result.lat - lat) > 0.0001 or abs(result.lon - lon) > 0.0001:
                    # Cache mismatch - delete stale entry
                    cache_path.unlink()
                    return None
                
                # Check TTL
                if time.time() - result.computed_at < self.CACHE_TTL:
                    return result
                else:
                    # Expired - delete
                    cache_path.unlink()
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, result: CachedResult):
        """Save result to cache."""
        cache_key = self._get_cache_key(result.lat, result.lon)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
    
    def get_garden_pins(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get front and back garden pins for a location.
        
        Optimized for speed - uses small radius and caching.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Use cached results if available
            
        Returns:
            Dict with 'front', 'back' pins and metadata
        """
        start_time = time.time()
        
        # Check fast cache (exact coordinate match)
        if use_cache:
            cached = self._load_from_cache(lat, lon)
            if cached:
                return {
                    "front": cached.front_pin,
                    "back": cached.back_pin,
                    "metadata": {
                        "from_cache": True,
                        "cache_source": "fast",
                        "processing_time": cached.processing_time,
                        "cached_at": cached.computed_at
                    }
                }
        
            # Check precompute cache (searches all cached areas at any zoom)
            try:
                manager = PrecomputeManager(tile_source=self.tile_source)
                pins = manager.get_nearest_building_pins(lat, lon)
                if pins["front"] is not None or pins["back"] is not None:
                    processing_time = time.time() - start_time
                    self._save_to_cache(CachedResult(
                        lat=lat, lon=lon,
                        front_pin=pins["front"],
                        back_pin=pins["back"],
                        classification=pins["front"].get("surface_type", "unknown") if pins["front"] else "unknown",
                        surface_type=pins["front"].get("surface_type", "unknown") if pins["front"] else "unknown",
                        score=pins["front"].get("score", 0) if pins["front"] else 0,
                        distance_to_building_m=pins["front"].get("distance_to_building_m", 0) if pins["front"] else 0,
                        computed_at=time.time(),
                        processing_time=processing_time
                    ))
                    return {
                        "front": pins["front"],
                        "back": pins["back"],
                        "metadata": {
                            "from_cache": True,
                            "cache_source": "precompute",
                            "processing_time": round(processing_time, 3)
                        }
                    }
            except Exception as e:
                print(f"    Precompute cache lookup error: {e}")
        
        # Process the location
        try:
            result = self._process_location(lat, lon)
            
            processing_time = time.time() - start_time
            
            # Cache the result
            cached_result = CachedResult(
                lat=lat,
                lon=lon,
                front_pin=result.get("front"),
                back_pin=result.get("back"),
                classification=result.get("classification", "unknown"),
                surface_type=result.get("surface_type", "unknown"),
                score=result.get("score", 0),
                distance_to_building_m=result.get("distance_to_building_m", 0),
                computed_at=time.time(),
                processing_time=processing_time
            )
            self._save_to_cache(cached_result)
            
            return {
                "front": result.get("front"),
                "back": result.get("back"),
                "metadata": {
                    "from_cache": False,
                    "processing_time": round(processing_time, 2)
                }
            }
            
        except Exception as e:
            return {
                "front": None,
                "back": None,
                "metadata": {
                    "error": str(e),
                    "processing_time": round(time.time() - start_time, 2)
                }
            }
    
    def classify_point(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Classify a single GPS coordinate.
        
        Returns classification, surface type, and score.
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(lat, lon)
            if cached:
                return {
                    "lat": lat,
                    "lon": lon,
                    "classification": cached.classification,
                    "surface_type": cached.surface_type,
                    "score": cached.score,
                    "distance_to_building_m": cached.distance_to_building_m,
                    "from_cache": True
                }
        
        # Process
        try:
            result = self._process_location(lat, lon)
            return {
                "lat": lat,
                "lon": lon,
                "classification": result.get("classification", "unknown"),
                "surface_type": result.get("surface_type", "unknown"),
                "score": result.get("score", 0),
                "distance_to_building_m": result.get("distance_to_building_m", 0),
                "from_cache": False
            }
        except Exception as e:
            return {
                "lat": lat,
                "lon": lon,
                "classification": "error",
                "error": str(e)
            }
    
    def _process_location(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Process a location to get garden classification and pins.
        
        Optimized for speed with minimal processing:
        - Small radius (50m)
        - Skip slow OSM queries (exclusion zones, property boundaries, addresses)
        - Use fast classification method
        """
        # Fetch imagery with small radius
        image, metadata = fetch_area_image(
            lat, lon,
            self.POINT_QUERY_RADIUS,
            self.zoom,
            show_progress=False,
            use_cache=True,
            tile_source=self.tile_source
        )
        
        if image is None:
            raise ValueError("Failed to fetch imagery")
        
        # Fetch essential OSM data (buildings, roads, driveways, boundaries)
        osm_data = self._fetch_essential_osm(lat, lon, self.POINT_QUERY_RADIUS)
        buildings = osm_data["buildings"]
        roads = osm_data["roads"]
        driveways = osm_data.get("driveways", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        property_boundaries = osm_data.get("property_boundaries", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
        
        if buildings.empty:
            return {
                "classification": "no_buildings",
                "surface_type": "unknown",
                "score": 0,
                "distance_to_building_m": 999,
                "front": None,
                "back": None
            }
        
        geo_bounds = metadata["geo_bounds"]
        image_size = tuple(metadata["image_size"])
        
        # Detect vegetation (optimized - skip enhanced detection)
        vegetation_mask, _ = detect_green_areas(image, show_progress=False, enhanced=False)
        
        # Convert to pixel coordinates
        building_polys_px = []
        for _, building in buildings.iterrows():
            coords = geometry_to_pixel_coords(building.geometry, geo_bounds, image_size)
            if coords:
                building_polys_px.append(coords)
        
        road_lines_px = []
        for _, road in roads.iterrows():
            coords = geometry_to_pixel_coords(road.geometry, geo_bounds, image_size)
            if coords:
                road_lines_px.append(coords)
        
        # Exclude buildings and roads
        vegetation_mask = exclude_buildings_from_mask(vegetation_mask, building_polys_px)
        vegetation_mask = exclude_roads_from_mask(vegetation_mask, road_lines_px, road_width_px=8)
        
        # Texture analysis: split vegetation into grass vs tree canopy
        # Uses CV (std/mean) + building proximity recovery
        mpp = (geo_bounds["east"] - geo_bounds["west"]) * 111320 * np.cos(np.radians(lat)) / image_size[0]
        grass_mask, tree_mask = split_vegetation_by_texture(
            image, vegetation_mask,
            building_polys_px=building_polys_px,
            meters_per_pixel=mpp
        )
        
        grass_mask = recover_shaded_grass(
            image, grass_mask, tree_mask,
            building_polys_px, road_lines_px,
            meters_per_pixel=mpp
        )
        
        # Classify gardens - include address data for accurate front/back detection
        address_polygons = osm_data.get("address_polygons", None)
        
        classifier = GardenClassifier(
            buildings=buildings,
            roads=roads,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=lat,
            center_lon=lon,
            driveways=driveways,
            exclusion_zones=None,  # Skip for speed
            property_boundaries=None,  # Skip for speed
            address_polygons=address_polygons  # Include for accurate front/back
        )
        
        # Create SPATIAL classification mask covering ALL potential garden areas
        # (not just grass) so paved gardens are also classified as front/back
        from scipy.ndimage import distance_transform_edt as edt_spatial
        import cv2 as cv2_spatial

        height_px, width_px = image_size[1], image_size[0]
        bld_mask_spatial = np.zeros((height_px, width_px), dtype=np.uint8)
        for poly in building_polys_px:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                cv2_spatial.fillPoly(bld_mask_spatial, [pts], 255)

        road_mask_spatial = np.zeros((height_px, width_px), dtype=np.uint8)
        for rline in road_lines_px:
            if len(rline) >= 2:
                pts = np.array(rline, dtype=np.int32)
                cv2_spatial.polylines(road_mask_spatial, [pts], False, 255, thickness=8)

        dist_to_bld_spatial = edt_spatial(bld_mask_spatial == 0) * mpp
        spatial_mask = np.zeros((height_px, width_px), dtype=np.uint8)
        spatial_mask[
            (dist_to_bld_spatial < 20) &
            (bld_mask_spatial == 0) &
            (road_mask_spatial == 0)
        ] = 255
        del bld_mask_spatial, road_mask_spatial, dist_to_bld_spatial

        classification_mask = classifier.classify_mask_fast(spatial_mask, show_progress=False)
        del spatial_mask
        
        # Find delivery pins
        pin_finder = DeliveryPinFinder(
            classification_mask=classification_mask,
            vegetation_mask=grass_mask,  # Only grass, not trees
            buildings=buildings,
            roads=roads,
            driveways=driveways,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=lat,
            center_lon=lon,
            property_boundaries=property_boundaries,
            tree_canopy_mask=tree_mask,  # For canopy density penalty
            original_vegetation_mask=vegetation_mask,  # Full veg mask before texture split
        )
        
        # Get pins for nearest building
        pins = pin_finder.find_nearest_building_pins(lat, lon)
        
        # Classify the specific point
        point_class = pin_finder.classify_point(lat, lon)
        
        return {
            "front": pins["front"].to_dict() if pins["front"] else None,
            "back": pins["back"].to_dict() if pins["back"] else None,
            "classification": point_class.get("classification", "unknown"),
            "surface_type": point_class.get("surface_type", "unknown"),
            "score": point_class.get("score", 0),
            "distance_to_building_m": point_class.get("distance_to_building_m", 0)
        }
    
    def _fetch_essential_osm(self, lat: float, lon: float, radius_m: float) -> Dict[str, Any]:
        """
        Fetch essential OSM data for point queries.
        
        Includes address_polygons (critical for accurate front/back detection)
        and property_boundaries (critical for preventing cross-property pins).
        
        Skips slow queries:
        - exclusion_zones (parks, pitches) - takes 1-6 minutes
        """
        # Check if we have cached essential data
        cache_key = get_osm_cache_key(lat, lon, radius_m)
        fast_cache_path = OSM_CACHE_DIR / f"{cache_key}_fast_v2.pkl"
        
        if fast_cache_path.exists():
            try:
                with open(fast_cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        
        # Fetch essential data (including addresses for front/back detection)
        buildings = fetch_buildings(lat, lon, radius_m, show_progress=False)
        roads = fetch_roads(lat, lon, radius_m, show_progress=False)
        driveways = fetch_driveways(lat, lon, radius_m, show_progress=False)
        
        # Address polygons tell us which street the building faces (critical!)
        try:
            address_polygons = fetch_address_polygons(lat, lon, radius_m, show_progress=False)
        except Exception:
            address_polygons = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        # Property boundaries (fences/walls/hedges) prevent cross-property pins
        try:
            property_boundaries = fetch_property_boundaries(lat, lon, radius_m, show_progress=False)
        except Exception:
            property_boundaries = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
        data = {
            "buildings": buildings,
            "roads": roads,
            "driveways": driveways,
            "address_polygons": address_polygons,
            "property_boundaries": property_boundaries,
        }
        
        # Cache for reuse
        try:
            OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(fast_cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass
        
        return data
    
    def clear_cache(self):
        """Clear the fast cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global classifier instance for reuse
_fast_classifier: Optional[FastGardenClassifier] = None


def get_fast_classifier(tile_source: TileSource = TileSource.AUTO) -> FastGardenClassifier:
    """Get or create the fast classifier singleton."""
    global _fast_classifier
    if _fast_classifier is None:
        _fast_classifier = FastGardenClassifier(tile_source=tile_source)
    return _fast_classifier
