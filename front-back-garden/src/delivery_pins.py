"""
Delivery Pin Scoring and Selection Module

Finds optimal delivery locations (pins) for front and back gardens.
Scoring prioritizes:
1. Grass (highest score)
2. Driveway near house (medium-high score)
3. Car parking area near house (medium score)

Ignores:
- Non-grass/non-driveway spaces
- Obstacles (for now - will be added later)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

import config
from src.osm import project_to_meters, geo_to_pixel


class SurfaceType(Enum):
    """Types of surfaces for delivery scoring."""
    UNKNOWN = 0
    GRASS = 1
    DRIVEWAY = 2
    PAVED = 3
    CAR_PARKING = 4
    BUILDING = 5
    ROAD = 6


@dataclass
class DeliveryPin:
    """
    Represents a potential delivery location.
    
    Attributes:
        lat: Latitude
        lon: Longitude  
        garden_type: "front" or "back"
        score: Delivery suitability score (0-100)
        surface_type: Type of surface at this location
        distance_to_building_m: Distance to nearest building in meters
        building_id: ID of the associated building (if any)
        metadata: Additional metadata
    """
    lat: float
    lon: float
    garden_type: str  # "front" or "back"
    score: float  # 0-100
    surface_type: str
    distance_to_building_m: float
    building_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Ensure all values are native Python types (not numpy)
        safe_metadata = {}
        for k, v in self.metadata.items():
            if hasattr(v, 'item'):  # numpy scalar
                safe_metadata[k] = v.item()
            else:
                safe_metadata[k] = v
        
        return {
            "lat": float(self.lat),
            "lon": float(self.lon),
            "garden_type": self.garden_type,
            "score": round(float(self.score), 2),
            "surface_type": self.surface_type,
            "distance_to_building_m": round(float(self.distance_to_building_m), 2),
            "building_id": self.building_id,
            "metadata": safe_metadata
        }


class DeliveryPinFinder:
    """
    Finds optimal delivery pin locations for front and back gardens.
    
    Scoring weights:
    - Grass: 100 (highest)
    - Driveway near house: 75
    - Paved area: 60
    - Car parking near house: 50
    
    Distance penalties:
    - Too close to building (<2m): -20
    - Too far from building (>20m): -10 per 5m over
    - Near road: -5
    
    Key validation:
    - Back garden pins must be BEHIND the building (opposite to road)
    - Pins must NOT be in front of any OTHER building (neighbor's front garden)
    """
    
    # Scoring weights for surface types
    SURFACE_SCORES = {
        SurfaceType.GRASS: 100,
        SurfaceType.DRIVEWAY: 75,
        SurfaceType.PAVED: 60,
        SurfaceType.CAR_PARKING: 50,
        SurfaceType.UNKNOWN: 30,
        SurfaceType.BUILDING: 0,
        SurfaceType.ROAD: 0,
    }
    
    # Optimal distance range from building (meters)
    OPTIMAL_DISTANCE_MIN = 2.0
    OPTIMAL_DISTANCE_MAX = 15.0
    
    def __init__(
        self,
        classification_mask: np.ndarray,
        vegetation_mask: np.ndarray,
        buildings: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame,
        driveways: gpd.GeoDataFrame,
        geo_bounds: dict,
        image_size: Tuple[int, int],
        center_lat: float,
        center_lon: float,
        meters_per_pixel: float = None,
        building_directions: Dict = None,
        property_boundaries: gpd.GeoDataFrame = None,
        tree_canopy_mask: np.ndarray = None,
        original_vegetation_mask: np.ndarray = None,
        building_polys_px: List = None,
        road_lines_px: List = None,
    ):
        """
        Initialize the delivery pin finder.
        
        Args:
            classification_mask: Classification mask (1=front, 2=back)
            vegetation_mask: Binary vegetation mask (grass only after texture split)
            buildings: GeoDataFrame with building polygons
            roads: GeoDataFrame with road geometries
            driveways: GeoDataFrame with driveway geometries
            geo_bounds: Geographic bounds dict
            image_size: (width, height) in pixels
            center_lat: Center latitude
            center_lon: Center longitude
            meters_per_pixel: Meters per pixel (calculated if not provided)
            building_directions: Dict mapping building idx to direction info (from GardenClassifier)
            property_boundaries: GeoDataFrame with fences/walls/hedges from OSM
            tree_canopy_mask: Binary mask where >0 = tree canopy (for density penalty)
            original_vegetation_mask: Full vegetation mask BEFORE texture split (grass+trees)
            building_polys_px: Pre-computed building polygon pixel coords (skip re-rasterization)
            road_lines_px: Pre-computed road line pixel coords (skip re-rasterization)
        """
        self.classification_mask = classification_mask
        self.vegetation_mask = vegetation_mask
        self.tree_canopy_mask = tree_canopy_mask
        self.original_vegetation_mask = original_vegetation_mask
        self.buildings = buildings
        self.roads = roads
        self.driveways = driveways
        self.property_boundaries = property_boundaries
        self.geo_bounds = geo_bounds
        self.image_size = image_size
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        # Calculate meters per pixel if not provided
        if meters_per_pixel is None:
            # Approximate using latitude
            lat_range = geo_bounds["north"] - geo_bounds["south"]
            meters_lat = lat_range * 111320  # ~111km per degree latitude
            self.meters_per_pixel = meters_lat / image_size[1]
        else:
            self.meters_per_pixel = meters_per_pixel
        
        # Store or compute building directions
        self.building_directions = building_directions or {}
        if not self.building_directions and not buildings.empty and not roads.empty:
            self._compute_building_directions()
        
        # Cache pre-computed pixel coords if provided (avoids re-rasterizing)
        self._building_polys_px = building_polys_px
        self._road_lines_px = road_lines_px
        
        # Pre-compute building and road masks
        self._precompute_masks()
        
        # Pre-compute distance transforms
        self._precompute_distances()
        
        # Track claimed garden areas to prevent multiple buildings pinning the same garden.
        # When a pin is placed, the surrounding area is marked as claimed.
        # The relaxed-ownership fallback (Attempt 3) excludes claimed areas.
        height, width = classification_mask.shape[:2] if classification_mask is not None else image_size[::-1]
        self._claimed_front = np.zeros((height, width), dtype=bool)
        self._claimed_back = np.zeros((height, width), dtype=bool)
    
    def _compute_building_directions(self):
        """Compute road-facing direction for each building."""
        from shapely.ops import nearest_points
        from src.osm import project_to_meters
        
        # Project to meters for accurate distance calculations
        buildings_m = project_to_meters(self.buildings, self.center_lat, self.center_lon)
        roads_m = project_to_meters(self.roads, self.center_lat, self.center_lon)
        
        if roads_m.empty:
            return
        
        # Combine all roads into one geometry
        from shapely.ops import unary_union
        all_roads = unary_union(roads_m.geometry.tolist())
        
        for idx, building in buildings_m.iterrows():
            try:
                centroid = building.geometry.centroid
                
                # Find nearest point on road
                nearest_road_point = nearest_points(centroid, all_roads)[1]
                
                # Direction vector from building to road (this is the "front" direction)
                dx = nearest_road_point.x - centroid.x
                dy = nearest_road_point.y - centroid.y
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    direction = (dx / length, dy / length)
                else:
                    direction = (0, 1)
                
                # Convert centroid to pixel coords
                bld_wgs = self.buildings.loc[idx]
                c = bld_wgs.geometry.centroid
                width, height = self.image_size
                cx = int((c.x - self.geo_bounds["west"]) / 
                        (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                cy = int((self.geo_bounds["north"] - c.y) / 
                        (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                
                self.building_directions[idx] = {
                    "centroid_px": (cx, cy),
                    "direction_to_road": direction,  # Unit vector pointing toward road
                    "distance_to_road": centroid.distance(nearest_road_point)
                }
            except Exception:
                continue
    
    def _precompute_masks(self):
        """Pre-compute building and road masks for fast lookup."""
        import time as _time
        from src.osm import geometry_to_pixel_coords
        
        t0 = _time.time()
        height, width = self.classification_mask.shape
        
        # Building mask -- use pre-computed pixel coords if available
        self.building_mask = np.zeros((height, width), dtype=np.uint8)
        if self._building_polys_px is not None:
            for poly in self._building_polys_px:
                if poly and len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(self.building_mask, [pts], 255)
        else:
            for _, building in self.buildings.iterrows():
                coords = geometry_to_pixel_coords(
                    building.geometry, self.geo_bounds, self.image_size
                )
                if coords and len(coords) >= 3:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.fillPoly(self.building_mask, [pts], 255)
        
        # Road mask -- use pre-computed pixel coords if available
        # Thickness covers the full road width (~6m = 20px at zoom 19, ~0.3m/px)
        road_thickness = max(8, int(6.0 / self.meters_per_pixel))
        self.road_mask = np.zeros((height, width), dtype=np.uint8)
        if self._road_lines_px is not None:
            for rline in self._road_lines_px:
                if rline and len(rline) >= 2:
                    pts = np.array(rline, dtype=np.int32)
                    cv2.polylines(self.road_mask, [pts], False, 255, thickness=road_thickness)
        else:
            for _, road in self.roads.iterrows():
                coords = geometry_to_pixel_coords(
                    road.geometry, self.geo_bounds, self.image_size
                )
                if coords and len(coords) >= 2:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.polylines(self.road_mask, [pts], False, 255, thickness=road_thickness)
        
        # Driveway mask
        self.driveway_mask = np.zeros((height, width), dtype=np.uint8)
        if self.driveways is not None and not self.driveways.empty:
            for _, driveway in self.driveways.iterrows():
                coords = geometry_to_pixel_coords(
                    driveway.geometry, self.geo_bounds, self.image_size
                )
                if coords and len(coords) >= 2:
                    pts = np.array(coords, dtype=np.int32)
                    cv2.polylines(self.driveway_mask, [pts], False, 255, thickness=6)
        
        print(f"    Rasterized masks ({len(self.buildings)} buildings, "
              f"{len(self.roads)} roads) ({_time.time()-t0:.1f}s)")
    
    def _precompute_distances(self):
        """Pre-compute distance transforms for buildings and roads."""
        import time as _time
        import gc as _gc
        t0 = _time.time()

        building_binary = self.building_mask > 0
        self.distance_to_building = distance_transform_edt(~building_binary).astype(np.float32)
        del building_binary
        _gc.collect()
        
        road_binary = self.road_mask > 0
        self.distance_to_road = distance_transform_edt(~road_binary).astype(np.float32)
        del road_binary
        _gc.collect()

        print(f"    Distance transforms ({_time.time()-t0:.1f}s)")
    
    def _precompute_vegetation_analysis(self):
        """
        Precompute vegetation statistics for graduated scoring.
        
        NOTE: Tree detection is now handled UPSTREAM via texture analysis in
        garden_detector.split_vegetation_by_texture(). The vegetation_mask
        passed to this class already contains only grass pixels (trees excluded).
        
        Creates:
        - _green_density_large: Grass pixel density in ~10m radius (float32)
        - _tree_mask: All-false mask (trees already filtered at source)
        - _veg_labels: Connected component labels for grass areas
        - _component_areas_m2: Area in m² per component label (1D, indexed by label)
        """
        from scipy.ndimage import uniform_filter
        
        veg_binary = (self.vegetation_mask > 0).astype(np.float32)
        
        # Grass density at large scale (~10m) for quality scoring
        self._green_density_large = uniform_filter(veg_binary, size=31).astype(np.float32)
        
        # Connected component analysis on grass-only mask
        veg_uint8 = (self.vegetation_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(veg_uint8, connectivity=8)
        
        self._veg_labels = labels
        
        # Per-component area in m²
        self._component_areas_m2 = stats[:, cv2.CC_STAT_AREA].astype(np.float32) * (self.meters_per_pixel ** 2)
        
        # No tree detection needed - trees were filtered out upstream by texture analysis.
        # Set tree mask to all-false for compatibility with downstream scoring code.
        self._is_tree_component = np.zeros(num_labels, dtype=bool)
        self._tree_mask = np.zeros_like(labels, dtype=bool)
        
        # Minimum area filter: grass components < 5m² are likely tree canopy fragments
        # or hedge overhangs, not real garden areas worth pinning
        self._large_enough = self._component_areas_m2 >= 5.0
        self._large_enough[0] = False  # Background label
        
        # Tree canopy density + tree fraction: skip for large images to save ~1GB.
        # The tree penalty is a scoring refinement, not essential for pin placement.
        height, width = self.vegetation_mask.shape
        large_image = (height * width) > 80_000_000  # ~8960x8960+

        if not large_image and self.tree_canopy_mask is not None:
            tree_binary = (self.tree_canopy_mask > 0).astype(np.float32)
            self._tree_canopy_density = uniform_filter(tree_binary, size=41).astype(np.float32)
            del tree_binary
        else:
            self._tree_canopy_density = np.zeros(1, dtype=np.float32)  # scalar placeholder
        
        if not large_image and self.original_vegetation_mask is not None and self.tree_canopy_mask is not None:
            orig_veg_binary = (self.original_vegetation_mask > 0).astype(np.float32)
            orig_veg_density = uniform_filter(orig_veg_binary, size=41).astype(np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                self._tree_fraction = np.where(
                    orig_veg_density > 0.05, self._tree_canopy_density / orig_veg_density, 0.0
                ).astype(np.float32)
            del orig_veg_binary, orig_veg_density
        else:
            self._tree_fraction = np.zeros(1, dtype=np.float32)  # scalar placeholder
        
        # Shade recovery: non-vegetation pixels that are likely grass in shadow.
        # When buildings cast shadow on front gardens (common in morning/evening
        # imagery), entire rows of gardens can be missed by green detection.
        #
        # Key improvement: compute vegetation ratio among GARDEN-LIKE pixels
        # only (exclude buildings and roads from the denominator).  A small
        # front garden squeezed between a building and a road would show near-
        # zero density with a naive average, but high density when only
        # garden-like pixels are counted.  A wider window (51px ≈ 15m) lets
        # the search reach sunlit back gardens behind the building.
        shade_window = 51 if not large_image else 31
        garden_area_f = ((self.building_mask == 0) & (self.road_mask == 0)).astype(np.float32)
        veg_in_garden = veg_binary * garden_area_f
        shade_num = uniform_filter(veg_in_garden, size=shade_window).astype(np.float32)
        shade_den = uniform_filter(garden_area_f, size=shade_window).astype(np.float32)
        del veg_in_garden, garden_area_f
        
        with np.errstate(divide='ignore', invalid='ignore'):
            shade_veg_ratio = np.where(
                shade_den > 0.01, shade_num / shade_den, 0.0
            ).astype(np.float32)
        del shade_num, shade_den
        
        max_bld_dist_px = 15.0 / self.meters_per_pixel
        self._shade_recovered = (
            (~(self.vegetation_mask > 0)) &
            (shade_veg_ratio > 0.15) &
            (self.distance_to_building < max_bld_dist_px) &
            (self.building_mask == 0) &
            (self.road_mask == 0) &
            (self.driveway_mask == 0)
        )
        del shade_veg_ratio
        shade_count = int(np.sum(self._shade_recovered))
        
        total_components = num_labels - 1
        large_components = int(np.sum(self._large_enough)) - (1 if self._large_enough[0] else 0)
        print(f"    Grass components: {total_components} total, {large_components} >= 5m² (trees filtered by texture)")
        if shade_count > 0:
            print(f"    Shade recovery: {shade_count:,} pixels recovered as grass")
    
    def _get_building_centroid_geo(self, building_idx: int) -> Tuple[float, float]:
        """Get geographic centroid (lat, lon) of a building by its list index."""
        building = self.buildings.iloc[building_idx]
        centroid = building.geometry.centroid
        return centroid.y, centroid.x
    
    def _precompute_property_zones(self):
        """
        Segment the image into property zones using physical boundaries.
        
        Uses buildings, roads, driveways, and fences/walls/hedges as barriers
        to create isolated property parcels. Each building is then mapped to
        its adjacent zones.
        
        A pin for building X can ONLY be placed within building X's zones.
        This prevents pins from crossing property boundaries into neighbors.
        
        Creates:
        - _property_zones: int32 array, zone label per pixel (0 = barrier)
        - _building_zones: dict mapping building_idx to set of adjacent zone labels
        """
        import time as _time
        from src.osm import geometry_to_pixel_coords
        
        t0 = _time.time()
        
        # Ensure building_owner is computed (needed for zone mapping)
        if not hasattr(self, 'building_owner'):
            self._precompute_building_zones()
        
        height, width = self.classification_mask.shape
        
        # Create barrier mask from all physical boundaries
        barrier = (self.building_mask > 0).astype(np.uint8) * 255
        barrier = np.maximum(barrier, self.road_mask)
        barrier = np.maximum(barrier, self.driveway_mask)
        
        # Add property boundaries (fences, walls, hedges) from OSM.
        # Barrier thickness derived from configured border width so that
        # the zone segmentation respects the inter-house border.
        boundary_thickness_px = max(3, int(config.PROPERTY_BORDER_WIDTH_M / self.meters_per_pixel))
        boundary_count = 0
        if self.property_boundaries is not None and not self.property_boundaries.empty:
            for _, boundary in self.property_boundaries.iterrows():
                geom = boundary.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    if geom.geom_type in ('LineString', 'MultiLineString'):
                        lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
                        for line in lines:
                            coords = geometry_to_pixel_coords(line, self.geo_bounds, self.image_size)
                            if coords and len(coords) >= 2:
                                pts = np.array(coords, dtype=np.int32)
                                cv2.polylines(barrier, [pts], False, 255, thickness=boundary_thickness_px)
                                boundary_count += 1
                    elif geom.geom_type in ('Polygon', 'MultiPolygon'):
                        coords = geometry_to_pixel_coords(geom, self.geo_bounds, self.image_size)
                        if coords and len(coords) >= 3:
                            pts = np.array(coords, dtype=np.int32)
                            cv2.polylines(barrier, [pts], True, 255, thickness=boundary_thickness_px)
                            boundary_count += 1
                except Exception:
                    continue
        
        # Label connected components of non-barrier pixels (connectivity=4 for strict separation)
        non_barrier = (barrier == 0).astype(np.uint8)
        num_zones, zone_labels = cv2.connectedComponents(non_barrier, connectivity=4)
        self._property_zones = zone_labels.astype(np.int32)
        
        # Map each building to its adjacent property zones (vectorized).
        # Old approach: per-building full-image scan = O(n_buildings * n_pixels).
        # New approach: extract valid (owner, zone) pairs once, group with numpy.
        close_to_building = (self.distance_to_building * self.meters_per_pixel) <= 3
        valid_mask = close_to_building & (zone_labels > 0) & (self.building_owner >= 0)
        
        owners_flat = self.building_owner[valid_mask].ravel()
        zones_flat = zone_labels[valid_mask].ravel()
        
        # Get unique (owner, zone) pairs and group by owner
        if len(owners_flat) > 0:
            pairs = np.unique(np.column_stack([owners_flat, zones_flat]), axis=0)
            self._building_zones = {i: set() for i in range(len(self.buildings))}
            for owner, zone in pairs:
                self._building_zones[int(owner)].add(int(zone))
        else:
            self._building_zones = {i: set() for i in range(len(self.buildings))}
        
        print(f"    Property zones: {num_zones} zones from {boundary_count} OSM boundaries ({_time.time()-t0:.1f}s)")
    
    def _precompute_building_zones(self):
        """
        FAST + LOW MEMORY: Precompute a per-pixel map of 'which building owns this pixel'.
        
        Uses a single labeled mask + one distance_transform_edt call with return_indices.
        Memory: ~3 arrays of image size, NOT per-building arrays.
        
        Result: self.building_owner[y, x] = building_list_index (or -1 if none).
        """
        import time as _time
        from src.osm import geometry_to_pixel_coords
        
        t0 = _time.time()
        height, width = self.classification_mask.shape
        
        # Create a labeled mask: each building gets its own label (idx + 1)
        labeled_mask = np.zeros((height, width), dtype=np.int16)
        self._building_centroids_px = {}
        
        for idx in range(len(self.buildings)):
            building = self.buildings.iloc[idx]
            coords = geometry_to_pixel_coords(building.geometry, self.geo_bounds, self.image_size)
            
            if not coords or len(coords) < 3:
                continue
            
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(labeled_mask, [pts], idx + 1)  # Label = idx + 1 (0 = no building)
            
            # Store centroid
            c = building.geometry.centroid
            cx = int((c.x - self.geo_bounds["west"]) / (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
            cy = int((self.geo_bounds["north"] - c.y) / (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
            self._building_centroids_px[idx] = (cx, cy)
        
        # Single distance transform: for every non-building pixel, find the nearest building pixel
        # return_indices gives us the coordinates of the nearest building pixel
        import gc as _gc
        is_background = (labeled_mask == 0)
        _, nearest_indices = distance_transform_edt(is_background, return_indices=True)
        del is_background
        
        # Look up the building label at the nearest indices, then free indices immediately
        nearest_labels = labeled_mask[nearest_indices[0], nearest_indices[1]]
        del nearest_indices, labeled_mask
        _gc.collect()
        
        # Convert labels back to building indices (label - 1), -1 for no building
        self.building_owner = (nearest_labels - 1).astype(np.int16)
        self.building_owner[nearest_labels == 0] = -1
        del nearest_labels
        _gc.collect()
        
        # Distance from each pixel to the nearest ownership boundary.
        # Used to push pins away from property edges (where hedges grow).
        owner_arr = self.building_owner
        edge_mask = np.zeros((height, width), dtype=np.uint8)
        edge_mask[1:, :] |= (owner_arr[1:, :] != owner_arr[:-1, :]).view(np.uint8)
        edge_mask[:, 1:] |= (owner_arr[:, 1:] != owner_arr[:, :-1]).view(np.uint8)
        self._owner_edge_dist = distance_transform_edt(edge_mask == 0).astype(np.float32)
        del edge_mask

        print(f"    Building ownership map ({len(self.buildings)} buildings) ({_time.time()-t0:.1f}s)")
    
    def _pixel_to_geo(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates."""
        width, height = self.image_size
        lon = self.geo_bounds["west"] + (px / width) * (self.geo_bounds["east"] - self.geo_bounds["west"])
        lat = self.geo_bounds["north"] - (py / height) * (self.geo_bounds["north"] - self.geo_bounds["south"])
        return lat, lon
    
    def _get_surface_type(self, px: int, py: int) -> SurfaceType:
        """
        Determine the surface type at a pixel location.
        
        IMPORTANT: Only returns GRASS if the vegetation mask confirms it.
        Non-vegetation areas get PAVED or DRIVEWAY scores (lower than grass).
        """
        # Check bounds
        height, width = self.classification_mask.shape
        if py < 0 or py >= height or px < 0 or px >= width:
            return SurfaceType.UNKNOWN
        
        if self.building_mask[py, px] > 0:
            return SurfaceType.BUILDING
        if self.road_mask[py, px] > 0:
            return SurfaceType.ROAD
        if self.driveway_mask[py, px] > 0:
            return SurfaceType.DRIVEWAY
        
        # CRITICAL: Only mark as GRASS if vegetation mask confirms it
        # This prevents non-grass front gardens from getting 100 score
        if self.vegetation_mask[py, px] > 0:
            return SurfaceType.GRASS
        
        # Shade recovery: pixel not in vegetation mask but surrounded by grass
        # is likely a shaded patch of the same garden, not paving
        if hasattr(self, '_shade_recovered') and self._shade_recovered[py, px]:
            return SurfaceType.GRASS
        
        # Non-vegetation areas - check if it might be a paved driveway/patio
        # by looking at distance to driveway or building
        dist_to_building = self.distance_to_building[py, px] * self.meters_per_pixel
        
        if dist_to_building < 5:
            return SurfaceType.CAR_PARKING
        
        return SurfaceType.PAVED
    
    def _calculate_score(
        self,
        px: int,
        py: int,
        surface_type: SurfaceType
    ) -> float:
        """
        Calculate delivery suitability score for a location.
        
        Scoring:
        - GRASS: 100 base (highest - best delivery surface)
        - DRIVEWAY: 75 base (good - near house, accessible)
        - PAVED: 60 base (okay - patios, concrete areas)
        - CAR_PARKING: 50 base (acceptable - might have cars)
        - UNKNOWN: 30 base (uncertain surface)
        - BUILDING/ROAD: 0 (not suitable)
        
        Args:
            px: Pixel X coordinate
            py: Pixel Y coordinate
            surface_type: Type of surface at this location
            
        Returns:
            Score from 0-100
        """
        height, width = self.classification_mask.shape
        if py < 0 or py >= height or px < 0 or px >= width:
            return 0
        
        # Base score from surface type - GRASS gets 100, others get less
        base_score = self.SURFACE_SCORES.get(surface_type, 30)
        
        # If not grass, the max score is capped at the surface type's base score
        # This ensures non-grass areas NEVER get 100
        if surface_type != SurfaceType.GRASS:
            # Non-grass areas are capped at their base score
            max_possible = base_score
        else:
            max_possible = 100
        
        # Distance to building penalty
        dist_building_px = self.distance_to_building[py, px]
        dist_building_m = dist_building_px * self.meters_per_pixel
        
        distance_penalty = 0
        if dist_building_m < self.OPTIMAL_DISTANCE_MIN:
            # Too close - penalty
            distance_penalty = -20
        elif dist_building_m > self.OPTIMAL_DISTANCE_MAX:
            # Too far - gradual penalty
            excess_m = dist_building_m - self.OPTIMAL_DISTANCE_MAX
            distance_penalty = -min(30, excess_m * 2)  # -2 per meter over, max -30
        
        # Distance to road penalty (don't want to be ON the road)
        dist_road_px = self.distance_to_road[py, px]
        dist_road_m = dist_road_px * self.meters_per_pixel
        
        road_penalty = 0
        if dist_road_m < 2:
            road_penalty = -15  # Too close to road
        
        # Calculate final score
        score = base_score + distance_penalty + road_penalty
        
        # Cap at max possible for this surface type
        score = min(score, max_possible)
        
        return max(0, min(100, score))
    
    def find_best_pin_for_building(
        self,
        building_idx: int,
        garden_type: str = "front",
        exclude_direction: Optional[Tuple[float, float]] = None
    ) -> Optional[DeliveryPin]:
        """
        Find the best delivery pin for a specific building.
        
        OPTIMIZED: Uses precomputed building_owner array instead of
        per-pixel Shapely distance calls.
        
        Args:
            building_idx: Index of the building in self.buildings
            garden_type: "front" or "back"
            exclude_direction: Optional (dx, dy) pixel-space direction vector.
                Candidates whose vector from building centroid has a positive
                dot product with this direction are excluded.  Used to force
                a pin to the opposite side of a previously placed pin.
            
        Returns:
            DeliveryPin or None if no suitable location found
        """
        if building_idx >= len(self.buildings):
            return None
        
        # Lazy-init precomputed data on first call.
        # Order matters for memory: property zones need building_owner but
        # NOT vegetation analysis.  Free large temporaries between steps.
        import gc as _gc_init
        if not hasattr(self, 'building_owner'):
            self._precompute_building_zones()
            _gc_init.collect()
        if not hasattr(self, '_property_zones'):
            self._precompute_property_zones()
            _gc_init.collect()
        if not hasattr(self, '_tree_mask'):
            self._precompute_vegetation_analysis()
            # Free source masks no longer needed (analysis extracted what it needs)
            self.tree_canopy_mask = None
            self.original_vegetation_mask = None
            _gc_init.collect()
        
        building = self.buildings.iloc[building_idx]
        building_idx_original = self.buildings.index[building_idx]
        
        # Skip very small buildings (sheds, garages, not houses)
        bounds = building.geometry.bounds
        lat_m = (bounds[3] - bounds[1]) * 111320
        lon_m = (bounds[2] - bounds[0]) * 111320 * np.cos(np.radians(self.center_lat))
        if lat_m * lon_m * 0.7 < 30:  # < ~30m²
            return None
        
        # Get centroid in pixel coordinates
        if building_idx in self._building_centroids_px:
            bld_cx, bld_cy = self._building_centroids_px[building_idx]
        else:
            c = building.geometry.centroid
            w, h = self.image_size
            bld_cx = int((c.x - self.geo_bounds["west"]) / (self.geo_bounds["east"] - self.geo_bounds["west"]) * w)
            bld_cy = int((self.geo_bounds["north"] - c.y) / (self.geo_bounds["north"] - self.geo_bounds["south"]) * h)
        
        target_class = 1 if garden_type == "front" else 2
        height, width = self.classification_mask.shape
        
        # Search radius (25m to catch full gardens including larger back gardens)
        search_radius_px = int(25 / self.meters_per_pixel)
        min_y = max(0, bld_cy - search_radius_px)
        max_y = min(height, bld_cy + search_radius_px)
        min_x = max(0, bld_cx - search_radius_px)
        max_x = min(width, bld_cx + search_radius_px)
        
        if max_y <= min_y or max_x <= min_x:
            return None
        
        # Extract the search region as numpy slices
        region_class = self.classification_mask[min_y:max_y, min_x:max_x]
        region_veg = self.vegetation_mask[min_y:max_y, min_x:max_x]
        region_bld = self.building_mask[min_y:max_y, min_x:max_x]
        region_road = self.road_mask[min_y:max_y, min_x:max_x]
        
        # Coordinate grids relative to building centroid (reused for distance + direction)
        ys_rel = np.arange(min_y, max_y).reshape(-1, 1) - bld_cy
        xs_rel = np.arange(min_x, max_x).reshape(1, -1) - bld_cx
        dist_to_centroid_m = np.sqrt(xs_rel**2 + ys_rel**2) * self.meters_per_pixel
        
        # Voronoi ownership + distance from building edge
        region_owner = self.building_owner[min_y:max_y, min_x:max_x]
        region_bld_dist = self.distance_to_building[min_y:max_y, min_x:max_x]
        max_garden_dist_m = 18.0  # Irish back gardens can be 15-20m deep
        
        # Exclude areas already claimed by another building's pin
        claimed_region = (self._claimed_front if garden_type == "front"
                          else self._claimed_back)[min_y:max_y, min_x:max_x]
        
        # Property border exclusion: hard constraint preventing pins from
        # being placed within the border zone between adjacent houses
        border_half_m = config.PROPERTY_BORDER_WIDTH_M / 2
        region_edge_dist_m = self._owner_edge_dist[min_y:max_y, min_x:max_x] * self.meters_per_pixel
        border_ok = region_edge_dist_m >= border_half_m
        
        # Lateral corridor constraint: restrict pins to within the building's
        # width (perpendicular to road direction) plus a small tolerance.
        # This prevents pins from drifting sideways into a neighbor's garden,
        # which the Voronoi ownership alone cannot reliably prevent.
        lateral_ok = np.ones((max_y - min_y, max_x - min_x), dtype=bool)
        dir_info = self.building_directions.get(building_idx_original)
        if dir_info is not None:
            road_dir = dir_info.get("direction_to_road")
            if road_dir is not None:
                try:
                    rd_x, rd_y = float(road_dir[0]), float(road_dir[1])
                    if rd_x * rd_x + rd_y * rd_y > 0.01:
                        lat_px_x, lat_px_y = -rd_y, -rd_x
                        
                        building_poly = self._geometry_to_pixel_polygon(building.geometry)
                        if building_poly is not None:
                            coords = list(building_poly.exterior.coords)
                            lateral_projs = [
                                (c[0] - bld_cx) * lat_px_x + (c[1] - bld_cy) * lat_px_y
                                for c in coords
                            ]
                            min_lat = min(lateral_projs)
                            max_lat = max(lateral_projs)
                            
                            corridor_tolerance_px = 3.0 / self.meters_per_pixel
                            min_lat -= corridor_tolerance_px
                            max_lat += corridor_tolerance_px
                            
                            pixel_lateral = xs_rel * lat_px_x + ys_rel * lat_px_y
                            lateral_ok = (pixel_lateral >= min_lat) & (pixel_lateral <= max_lat)
                except Exception:
                    pass
        
        # Shade recovery region (for grass detection in shaded areas)
        region_shade = self._shade_recovered[min_y:max_y, min_x:max_x]
        
        # Base spatial filters: not on building/road, within distance,
        # owned by this building, not already claimed, not in border zone,
        # and within the building's lateral corridor
        base_spatial = (
            (region_bld == 0) &
            (region_road == 0) &
            (region_owner == building_idx) &
            (region_bld_dist * self.meters_per_pixel <= max_garden_dist_m) &
            (~claimed_region) &
            border_ok &
            lateral_ok
        )
        
        # Opposite-side enforcement: exclude the half-plane containing
        # a previously placed pin so the new pin lands on the other side.
        if exclude_direction is not None:
            ed_x, ed_y = float(exclude_direction[0]), float(exclude_direction[1])
            dot_map = xs_rel * ed_x + ys_rel * ed_y
            base_spatial = base_spatial & (dot_map <= 0)
        
        # Property zone constraint
        property_mask = np.ones_like(base_spatial)
        if building_idx in self._building_zones:
            valid_zones = self._building_zones[building_idx]
            if valid_zones:
                region_zones = self._property_zones[min_y:max_y, min_x:max_x]
                property_mask = np.isin(region_zones, list(valid_zones))
            else:
                return None
        
        # Road-side vegetation filter: remove vegetation that is closer to
        # a road than to the building — likely street trees, hedge borders
        # along footpaths, or roadside planting rather than garden grass.
        # A minimum building distance of 4m protects front garden grass
        # that is naturally between the house and the road.
        region_road_dist = self.distance_to_road[min_y:max_y, min_x:max_x]
        road_dist_m = region_road_dist * self.meters_per_pixel
        bld_dist_m = region_bld_dist * self.meters_per_pixel
        road_veg_filter = ~(
            (region_veg > 0) &
            (road_dist_m < bld_dist_m) &
            (bld_dist_m > 4)
        )
        
        # ---- FALLBACK SYSTEM (tracks which attempt succeeds) ----
        # Classification mask has per-pixel front/back from road geometry.
        # No per-building direction filter -- trust the mask.
        region_tree = self._tree_mask[min_y:max_y, min_x:max_x]
        region_density_large = self._green_density_large[min_y:max_y, min_x:max_x]
        region_labels = self._veg_labels[min_y:max_y, min_x:max_x]
        attempt_used = 0
        
        # Attempt 1: Classification mask + ownership + property zone (strictest)
        candidates = (base_spatial & (region_class == target_class) &
                      property_mask & road_veg_filter)
        if np.any(candidates):
            attempt_used = 1
        
        # Attempt 2: Classification mask + ownership only (drop property zone)
        if attempt_used == 0:
            candidates = (base_spatial & (region_class == target_class))
            if np.any(candidates):
                attempt_used = 2
        
        # Attempt 3: Ownership only (drop class mask -- no front/back preference)
        if attempt_used == 0:
            candidates = (base_spatial & property_mask)
            if np.any(candidates):
                attempt_used = 3
        
        # Within the selected candidates, prefer grass over paved -- BUT only
        # grass that is well within this property.  Grass near the ownership
        # boundary is likely the neighbor's garden edge leaking in via the
        # corridor tolerance.  Requiring a stricter edge distance for grass
        # preference means paved gardens get their own pin instead of
        # stealing the neighbor's grass.
        grass_interior_m = config.PROPERTY_BORDER_WIDTH_M * 1.5
        grass_interior_ok = region_edge_dist_m >= grass_interior_m
        
        grass_candidates = (candidates &
                            ((region_veg > 0) | region_shade) &
                            (~region_tree) &
                            grass_interior_ok)
        if np.any(grass_candidates):
            large_enough = self._large_enough[region_labels]
            quality_grass = grass_candidates & large_enough & (region_density_large > 0.1)
            
            if np.any(quality_grass):
                candidates = quality_grass
            else:
                basic_grass = grass_candidates & large_enough
                if np.any(basic_grass):
                    candidates = basic_grass
                elif np.any(grass_candidates):
                    candidates = grass_candidates
                # else: no interior grass → all candidates remain, scoring
                # picks the best paved/driveway location for this property
        
        # Find best scoring candidate
        if not np.any(candidates):
            return None
        
        # Score all candidates at once
        candidate_ys, candidate_xs = np.where(candidates)
        
        if len(candidate_ys) == 0:
            return None
        
        # Graduated scoring based on surface type, garden quality, and distance
        scores = np.full(len(candidate_ys), 35.0)
        for i, (cy_r, cx_r) in enumerate(zip(candidate_ys, candidate_xs)):
            py_abs = min_y + cy_r
            px_abs = min_x + cx_r
            
            is_shade_recovered = self._shade_recovered[py_abs, px_abs]
            is_grass = ((self.vegetation_mask[py_abs, px_abs] > 0 and
                        not self._tree_mask[py_abs, px_abs]) or
                       is_shade_recovered)
            is_driveway = self.driveway_mask[py_abs, px_abs] > 0
            
            if is_grass:
                # Grass base score: 70
                base = 70.0
                
                # Green density bonus (0-15): larger contiguous green = better garden
                density = float(self._green_density_large[py_abs, px_abs])
                density_bonus = min(15.0, density * 25.0)
                
                # Component area bonus (0-10): bigger garden = higher score
                comp_label = self._veg_labels[py_abs, px_abs]
                comp_area = float(self._component_areas_m2[comp_label]) if comp_label > 0 else 0.0
                area_bonus = min(10.0, comp_area / 5.0)  # 50m² for full bonus
                
                scores[i] = base + density_bonus + area_bonus
            elif is_driveway:
                scores[i] = 60.0
            else:
                scores[i] = 40.0
            
            # Distance adjustment
            dist_m = self.distance_to_building[py_abs, px_abs] * self.meters_per_pixel
            if dist_m < 2:
                scores[i] -= 15
            elif dist_m < 5:
                scores[i] += 5
            elif dist_m < 10:
                scores[i] += 2
            elif dist_m > 15:
                scores[i] -= min(20, (dist_m - 15) * 3)
            
            # Road proximity penalty
            dist_road = self.distance_to_road[py_abs, px_abs] * self.meters_per_pixel
            if dist_road < 2:
                scores[i] -= 10
            
            # Property boundary penalty: graduated score reduction for pins
            # approaching the border zone (hard exclusion is in base_spatial,
            # this adds a softer falloff beyond the hard border)
            edge_dist_m = self._owner_edge_dist[py_abs, px_abs] * self.meters_per_pixel
            soft_border_m = config.PROPERTY_BORDER_WIDTH_M + 1.0
            if edge_dist_m < soft_border_m:
                scores[i] -= min(20, (soft_border_m - edge_dist_m) * 7)
            
            # Tree canopy penalty: ONLY for grass candidates.
            # Skipped for large images (arrays are scalar placeholders).
            if is_grass and self._tree_canopy_density.ndim > 1:
                tree_density = float(self._tree_canopy_density[py_abs, px_abs])
                tree_frac = float(self._tree_fraction[py_abs, px_abs])
                
                tree_penalty = 0.0
                if tree_density > 0.20:
                    tree_penalty += min(35, (tree_density - 0.20) * 70)
                if tree_frac > 0.4:
                    tree_penalty += min(25, (tree_frac - 0.4) * 50)
                
                if tree_penalty > 0:
                    scores[i] -= min(50, tree_penalty)
            
            # Cap by surface type
            if is_grass:
                scores[i] = max(0, min(100, scores[i]))
            elif is_driveway:
                scores[i] = max(0, min(75, scores[i]))
            else:
                scores[i] = max(0, min(60, scores[i]))
        
        # CENTROID-BASED placement: put pin in center of garden, not at edge
        max_score = np.max(scores)
        if max_score <= 0:
            return None
        
        # Select top-scoring candidates (within 50% of max) to define garden area
        good_mask = scores >= max_score * 0.5
        good_ys = candidate_ys[good_mask]
        good_xs = candidate_xs[good_mask]
        good_scores = scores[good_mask]
        
        # Compute centroid of the good candidates
        centroid_y = np.mean(good_ys)
        centroid_x = np.mean(good_xs)
        
        # Find candidate nearest to centroid (centers pin in the garden)
        dists_to_center = (good_ys - centroid_y)**2 + (good_xs - centroid_x)**2
        nearest_idx = np.argmin(dists_to_center)
        
        best_py = min_y + good_ys[nearest_idx]
        best_px = min_x + good_xs[nearest_idx]
        best_score = good_scores[nearest_idx]
        
        if best_score <= 0:
            return None
        
        # Mark garden area around pin as CLAIMED to prevent other buildings
        # from placing their pin in the same garden (prevents clustering).
        claim_r = int(5.0 / self.meters_per_pixel)
        h_img, w_img = self.classification_mask.shape
        cy1 = max(0, best_py - claim_r)
        cy2 = min(h_img, best_py + claim_r + 1)
        cx1 = max(0, best_px - claim_r)
        cx2 = min(w_img, best_px + claim_r + 1)
        claimed = self._claimed_front if garden_type == "front" else self._claimed_back
        yy, xx = np.ogrid[cy1-best_py:cy2-best_py, cx1-best_px:cx2-best_px]
        circle = (yy**2 + xx**2) <= claim_r**2
        claimed[cy1:cy2, cx1:cx2] |= circle
        
        # Determine surface type (including shade-recovered grass)
        if self.vegetation_mask[best_py, best_px] > 0:
            if self._tree_mask[best_py, best_px]:
                surface = "tree"
            else:
                surface = "grass"
        elif self._shade_recovered[best_py, best_px]:
            surface = "grass"
        elif self.driveway_mask[best_py, best_px] > 0:
            surface = "driveway"
        else:
            surface = "paved"
        
        lat, lon = self._pixel_to_geo(best_px, best_py)
        dist_m = self.distance_to_building[best_py, best_px] * self.meters_per_pixel
        
        building_id = None
        if "osm_id" in building.index:
            building_id = str(building["osm_id"])
        elif hasattr(building, 'name') and building.name is not None:
            building_id = str(building.name)
        
        return DeliveryPin(
            lat=lat, lon=lon,
            garden_type=garden_type,
            score=round(max(0, min(100, best_score)), 2),
            surface_type=surface,
            distance_to_building_m=round(dist_m, 2),
            building_id=building_id,
            metadata={"pixel_x": best_px, "pixel_y": best_py, "attempt": attempt_used}
        )
    
    def find_all_pins(
        self,
        include_front: bool = True,
        include_back: bool = True,
        min_score: float = 0.0,
        show_progress: bool = True,
        include_no_garden: bool = True
    ) -> List[DeliveryPin]:
        """
        Find delivery pins for all buildings.
        
        Args:
            include_front: Include front garden pins
            include_back: Include back garden pins
            min_score: Minimum score threshold
            show_progress: Show tqdm progress bar
            include_no_garden: Include "no_garden" entries for buildings without suitable garden
            
        Returns:
            List of DeliveryPin objects
        """
        from tqdm import tqdm
        
        # Precompute building zones once (fast numpy operation)
        if not hasattr(self, 'building_owner'):
            self._precompute_building_zones()
        
        pins = []
        iterator = range(len(self.buildings))
        if show_progress:
            iterator = tqdm(iterator, desc="Finding pins", unit="bld")
        
        for idx in iterator:
            building = self.buildings.iloc[idx]
            building_id = None
            if "osm_id" in building.index:
                building_id = str(building["osm_id"])
            elif hasattr(building, 'name') and building.name is not None:
                building_id = str(building.name)
            else:
                building_id = f"bld_{idx}"
            
            if include_front:
                front_pin = self.find_best_pin_for_building(idx, "front")
                if front_pin and front_pin.score >= min_score:
                    pins.append(front_pin)
                elif include_no_garden:
                    lat, lon = self._get_building_centroid_geo(idx)
                    pins.append(DeliveryPin(
                        lat=lat, lon=lon,
                        garden_type="front", score=0.0,
                        surface_type="no_garden",
                        distance_to_building_m=0.0,
                        building_id=building_id,
                        metadata={"pixel_x": 0, "pixel_y": 0}
                    ))
            
            if include_back:
                back_pin = self.find_best_pin_for_building(idx, "back")
                if back_pin and back_pin.score >= min_score:
                    pins.append(back_pin)
                elif include_no_garden:
                    lat, lon = self._get_building_centroid_geo(idx)
                    pins.append(DeliveryPin(
                        lat=lat, lon=lon,
                        garden_type="back", score=0.0,
                        surface_type="no_garden",
                        distance_to_building_m=0.0,
                        building_id=building_id,
                        metadata={"pixel_x": 0, "pixel_y": 0}
                    ))
        
        return pins
    
    def classify_point(self, lat: float, lon: float) -> dict:
        """
        Classify a single GPS coordinate as front or back garden.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict with classification result
        """
        width, height = self.image_size
        
        # Convert to pixel coordinates
        px = int((lon - self.geo_bounds["west"]) / 
                (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
        py = int((self.geo_bounds["north"] - lat) / 
                (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
        
        # Check bounds
        if px < 0 or px >= width or py < 0 or py >= height:
            return {
                "lat": lat,
                "lon": lon,
                "classification": "out_of_bounds",
                "confidence": 0.0
            }
        
        # Get classification
        class_value = self.classification_mask[py, px]
        
        if class_value == 1:
            classification = "front_garden"
        elif class_value == 2:
            classification = "back_garden"
        elif class_value == 3:
            classification = "unknown"
        else:
            # Check if it's a building, road, etc.
            surface_type = self._get_surface_type(px, py)
            classification = surface_type.name.lower()
        
        # Get surface type and score
        surface_type = self._get_surface_type(px, py)
        score = self._calculate_score(px, py, surface_type)
        
        return {
            "lat": lat,
            "lon": lon,
            "classification": classification,
            "surface_type": surface_type.name.lower(),
            "score": round(score, 2),
            "distance_to_building_m": round(
                self.distance_to_building[py, px] * self.meters_per_pixel, 2
            )
        }
    
    def find_nearest_building_pins(
        self,
        lat: float,
        lon: float,
        max_distance_m: float = 50.0
    ) -> Dict[str, Optional[DeliveryPin]]:
        """
        Find the best front and back garden pins for the building at/nearest to a coordinate.
        
        IMPROVED: First checks if the point is INSIDE or very close to a building.
        Only searches for gardens that are directly adjacent to THAT building.
        
        Args:
            lat: Latitude
            lon: Longitude
            max_distance_m: Maximum distance to search for buildings
            
        Returns:
            Dict with 'front' and 'back' DeliveryPin objects (or None if not found)
        """
        if self.buildings.empty:
            return {"front": None, "back": None}
        
        # Convert input point to pixel coordinates
        width, height = self.image_size
        input_px = int((lon - self.geo_bounds["west"]) / 
                      (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
        input_py = int((self.geo_bounds["north"] - lat) / 
                      (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
        
        # Project buildings to meters for distance calculation
        buildings_m = project_to_meters(self.buildings, self.center_lat, self.center_lon)
        
        # Create point and project it
        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )
        point_m = project_to_meters(point, self.center_lat, self.center_lon)
        point_geom = point_m.geometry.iloc[0]
        
        # First: Check if point is INSIDE a building
        containing_idx = None
        for idx, building in buildings_m.iterrows():
            if building.geometry.contains(point_geom) or building.geometry.distance(point_geom) < 3:
                containing_idx = idx
                break
        
        # If point is inside a building, use that building
        if containing_idx is not None:
            nearest_idx = containing_idx
            min_dist = 0
        else:
            # Find nearest building
            min_dist = float("inf")
            nearest_idx = None
            
            for idx, building in buildings_m.iterrows():
                dist = point_geom.distance(building.geometry)
                if dist < min_dist and dist < max_distance_m:
                    min_dist = dist
                    nearest_idx = idx
        
        if nearest_idx is None:
            return {"front": None, "back": None}
        
        # Get the building in WGS84
        building_wgs = self.buildings.loc[nearest_idx]
        building_centroid = building_wgs.geometry.centroid
        
        # Convert building centroid to pixel coordinates
        bld_cx = int((building_centroid.x - self.geo_bounds["west"]) / 
                    (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
        bld_cy = int((self.geo_bounds["north"] - building_centroid.y) / 
                    (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
        
        # Find the index in the original DataFrame
        building_list_idx = list(self.buildings.index).index(nearest_idx)
        
        # Find front and back pins for THIS specific building
        front_pin = self._find_pin_for_specific_building(
            building_list_idx, bld_cx, bld_cy, input_px, input_py, "front"
        )
        back_pin = self._find_pin_for_specific_building(
            building_list_idx, bld_cx, bld_cy, input_px, input_py, "back"
        )
        
        # Same-side correction: if both pins are <8m apart, re-search
        # the lower-scored pin on the opposite side of the building,
        # then assign labels by road distance (front = closer to road).
        if front_pin and back_pin and front_pin.score > 0 and back_pin.score > 0:
            sep_m = np.sqrt(
                ((front_pin.lat - back_pin.lat) * 111320) ** 2
                + ((front_pin.lon - back_pin.lon) * 111320 * np.cos(np.radians(self.center_lat))) ** 2
            )
            if sep_m < 8:
                keep, redo_type = (front_pin, "back") if front_pin.score >= back_pin.score else (back_pin, "front")
                keep_px_meta = keep.metadata.get("pixel_x", None)
                keep_py_meta = keep.metadata.get("pixel_y", None)
                if keep_px_meta is not None and keep_py_meta is not None:
                    exclude_dir = (int(keep_px_meta) - bld_cx, int(keep_py_meta) - bld_cy)
                    if abs(exclude_dir[0]) >= 1 or abs(exclude_dir[1]) >= 1:
                        new_pin = self.find_best_pin_for_building(
                            building_list_idx, redo_type, exclude_direction=exclude_dir
                        )
                        if new_pin and new_pin.score > 0:
                            if redo_type == "front":
                                front_pin = new_pin
                            else:
                                back_pin = new_pin

                        # Assign front/back by road distance
                        fp_meta = front_pin.metadata.get("pixel_x"), front_pin.metadata.get("pixel_y")
                        bp_meta = back_pin.metadata.get("pixel_x"), back_pin.metadata.get("pixel_y")
                        if fp_meta[0] is not None and bp_meta[0] is not None:
                            h, w = self.classification_mask.shape
                            fpx = max(0, min(w - 1, int(fp_meta[0])))
                            fpy = max(0, min(h - 1, int(fp_meta[1])))
                            bpx = max(0, min(w - 1, int(bp_meta[0])))
                            bpy = max(0, min(h - 1, int(bp_meta[1])))
                            f_road = float(self.distance_to_road[fpy, fpx])
                            b_road = float(self.distance_to_road[bpy, bpx])
                            if f_road > b_road * 1.05:
                                front_pin, back_pin = back_pin, front_pin
                                front_pin.garden_type = "front"
                                back_pin.garden_type = "back"
        
        return {
            "front": front_pin,
            "back": back_pin
        }
    
    def _find_pin_for_specific_building(
        self,
        building_idx: int,
        bld_cx: int,
        bld_cy: int,
        input_px: int,
        input_py: int,
        garden_type: str
    ) -> Optional[DeliveryPin]:
        """
        Find a pin for a specific building, ensuring it's in the correct garden.
        
        Key improvements:
        1. Creates a Voronoi-style zone for this building
        2. Only considers pixels closest to THIS building
        3. Filters out isolated green (likely trees) - must be connected to building-adjacent area
        4. Requires green area to be contiguous with building edge
        """
        if building_idx >= len(self.buildings):
            return None
        
        building = self.buildings.iloc[building_idx]
        target_class = 1 if garden_type == "front" else 2
        
        height, width = self.classification_mask.shape
        
        # Build a map of "closest building" for each pixel (Voronoi-style)
        building_zone_mask = self._get_building_zone_mask(building_idx, bld_cx, bld_cy)
        
        # Search radius in pixels (about 15 meters - tighter to avoid neighbor gardens)
        search_radius_px = int(15 / self.meters_per_pixel)
        
        min_y = max(0, bld_cy - search_radius_px)
        max_y = min(height, bld_cy + search_radius_px)
        min_x = max(0, bld_cx - search_radius_px)
        max_x = min(width, bld_cx + search_radius_px)
        
        best_score = -1
        best_px, best_py = None, None
        best_surface = None
        
        # Get target building polygon for additional checks
        target_building_poly = self._geometry_to_pixel_polygon(building.geometry)
        
        # Get our building's direction info
        building_idx_original = self.buildings.index[building_idx]
        our_direction_info = self.building_directions.get(building_idx_original)
        
        # Sample every 2 pixels for better accuracy
        for py in range(min_y, max_y, 2):
            for px in range(min_x, max_x, 2):
                # CRITICAL: Must be in THIS building's zone
                if not building_zone_mask[py, px]:
                    continue
                
                # Property border enforcement: skip pixels too close to
                # the ownership boundary between adjacent houses
                if hasattr(self, '_owner_edge_dist'):
                    edge_dist_m = self._owner_edge_dist[py, px] * self.meters_per_pixel
                    if edge_dist_m < config.PROPERTY_BORDER_WIDTH_M / 2:
                        continue
                
                # Must be correct garden type
                if self.classification_mask[py, px] != target_class:
                    continue
                
                # DIRECTIONAL CHECK: Verify pixel is on correct side of building
                if our_direction_info and not self._is_correct_side(
                    px, py, bld_cx, bld_cy, our_direction_info, garden_type
                ):
                    continue
                
                # LATERAL ALIGNMENT CHECK: pixel must be within the building's
                # width corridor, not shifted sideways into neighbor's garden
                if not self._is_laterally_aligned(
                    px, py, building_idx, garden_type
                ):
                    continue
                
                # NEIGHBOR CHECK: Ensure this isn't in front of another building
                if self._is_front_of_other_building(px, py, building_idx_original, garden_type):
                    continue
                
                # Distance to our building in meters (using the distance transform)
                dist_to_any_building_m = self.distance_to_building[py, px] * self.meters_per_pixel
                
                # Also check distance to OUR specific building
                if target_building_poly:
                    point = Point(px, py)
                    dist_to_our_building_px = point.distance(target_building_poly)
                    dist_to_our_building_m = dist_to_our_building_px * self.meters_per_pixel
                else:
                    dist_to_our_building_m = dist_to_any_building_m
                
                # Skip if too far from OUR building (stricter - 10m max)
                if dist_to_our_building_m > 10:
                    continue
                
                # Get surface type
                surface_type = self._get_surface_type(px, py)
                
                # Skip buildings and roads
                if surface_type in (SurfaceType.BUILDING, SurfaceType.ROAD):
                    continue
                
                # For grass, verify it's a substantial area (not an isolated tree)
                if surface_type == SurfaceType.GRASS:
                    if not self._is_valid_grass_area(px, py):
                        continue
                
                # Calculate score
                score = self._calculate_score(px, py, surface_type)
                
                if score > best_score:
                    best_score = score
                    best_px, best_py = px, py
                    best_surface = surface_type
        
        if best_px is None:
            # Try fallback with driveway/paved (non-grass) in this zone
            return self._find_non_grass_pin(building_idx, bld_cx, bld_cy, building_zone_mask, garden_type)
        
        # Convert to geographic coordinates
        lat, lon = self._pixel_to_geo(best_px, best_py)
        
        # Calculate distance to building
        dist_m = self.distance_to_building[best_py, best_px] * self.meters_per_pixel
        
        # Get building ID
        building_id = None
        if "osm_id" in building.index:
            building_id = str(building["osm_id"])
        elif hasattr(building, 'name') and building.name is not None:
            building_id = str(building.name)
        
        return DeliveryPin(
            lat=lat,
            lon=lon,
            garden_type=garden_type,
            score=best_score,
            surface_type=best_surface.name.lower(),
            distance_to_building_m=dist_m,
            building_id=building_id,
            metadata={
                "pixel_x": best_px,
                "pixel_y": best_py,
            }
        )
    
    def _get_building_zone_mask(self, target_idx: int, target_cx: int, target_cy: int) -> np.ndarray:
        """
        Create a mask where pixels are True ONLY if they're in the target building's garden zone.
        
        Approach:
        1. Pixel must be closer to target building than ANY other building
        2. Pixel must NOT be between another building and its likely garden area
        3. Uses building edges, not just centroids, for accurate distance calculation
        """
        height, width = self.classification_mask.shape
        mask = np.zeros((height, width), dtype=bool)  # Start with all False
        
        # Get target building geometry in pixel coords
        target_building = self.buildings.iloc[target_idx]
        target_poly_px = self._geometry_to_pixel_polygon(target_building.geometry)
        
        # Get all other building polygons in pixel coords
        other_buildings_px = []
        for idx, bld in self.buildings.iterrows():
            if list(self.buildings.index).index(idx) == target_idx:
                continue
            poly_px = self._geometry_to_pixel_polygon(bld.geometry)
            if poly_px is not None:
                c = bld.geometry.centroid
                cx = int((c.x - self.geo_bounds["west"]) / 
                        (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                cy = int((self.geo_bounds["north"] - c.y) / 
                        (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
                other_buildings_px.append((poly_px, cx, cy))
        
        # Search radius - only consider pixels close to our building
        search_radius = int(20 / self.meters_per_pixel)
        min_y = max(0, target_cy - search_radius)
        max_y = min(height, target_cy + search_radius)
        min_x = max(0, target_cx - search_radius)
        max_x = min(width, target_cx + search_radius)
        
        for py in range(min_y, max_y):
            for px in range(min_x, max_x):
                # Distance to target building EDGE (not centroid)
                point = Point(px, py)
                dist_to_target = point.distance(target_poly_px) if target_poly_px else float('inf')
                
                # Check if this pixel is closer to our building than others
                is_closest_to_target = True
                for other_poly, other_cx, other_cy in other_buildings_px:
                    dist_to_other = point.distance(other_poly)
                    
                    # Must be significantly closer to target (1.5x rule)
                    # This prevents selecting areas that are ambiguously between buildings
                    if dist_to_other < dist_to_target * 1.5:
                        is_closest_to_target = False
                        break
                
                if is_closest_to_target:
                    mask[py, px] = True
        
        return mask
    
    def _geometry_to_pixel_polygon(self, geom) -> Optional[Polygon]:
        """Convert a shapely geometry to pixel coordinates."""
        from shapely.geometry import Polygon, MultiPolygon
        
        width, height = self.image_size
        
        def transform_coords(coords):
            result = []
            for lon, lat in coords:
                px = (lon - self.geo_bounds["west"]) / (self.geo_bounds["east"] - self.geo_bounds["west"]) * width
                py = (self.geo_bounds["north"] - lat) / (self.geo_bounds["north"] - self.geo_bounds["south"]) * height
                result.append((px, py))
            return result
        
        try:
            if geom.geom_type == 'Polygon':
                return Polygon(transform_coords(geom.exterior.coords))
            elif geom.geom_type == 'MultiPolygon':
                # Return the largest polygon
                largest = max(geom.geoms, key=lambda p: p.area)
                return Polygon(transform_coords(largest.exterior.coords))
            else:
                return None
        except Exception:
            return None
    
    def _is_valid_grass_area(self, px: int, py: int, min_connected_pixels: int = 50) -> bool:
        """
        Check if a grass pixel is part of a substantial grass area.
        
        Filters out isolated trees which appear as small circular green blobs.
        Valid grass should be:
        - Part of a larger contiguous area
        - Connected to building-adjacent areas
        """
        height, width = self.vegetation_mask.shape
        
        # Quick check: count vegetation pixels in small neighborhood
        radius = 5
        min_y = max(0, py - radius)
        max_y = min(height, py + radius)
        min_x = max(0, px - radius)
        max_x = min(width, px + radius)
        
        neighborhood = self.vegetation_mask[min_y:max_y, min_x:max_x]
        green_count = np.sum(neighborhood > 0)
        
        # If very few green pixels nearby, likely isolated tree
        if green_count < 20:
            return False
        
        # Check if building is nearby (grass should be adjacent to a building)
        building_dist = self.distance_to_building[py, px] * self.meters_per_pixel
        if building_dist > 15:
            return False
        
        return True
    
    def _is_correct_side(
        self,
        px: int,
        py: int,
        bld_cx: int,
        bld_cy: int,
        direction_info: Dict,
        garden_type: str
    ) -> bool:
        """
        Check if pixel is on the correct side of the building for the garden type.
        
        - Front garden: pixel should be in the direction of the road
        - Back garden: pixel should be OPPOSITE to the road direction
        """
        road_dir = direction_info.get("direction_to_road")
        if road_dir is None:
            return True  # No direction info, allow it
        
        # Ensure road_dir is indexable (could be tuple, list, or numpy array)
        try:
            rd_x = float(road_dir[0])
            rd_y = float(road_dir[1])
        except (IndexError, TypeError, ValueError):
            return True
        
        # Vector from building centroid to pixel
        dx = px - bld_cx
        dy = py - bld_cy
        length = np.sqrt(dx**2 + dy**2)
        
        if length < 1:
            return True  # Too close to centroid, allow it
        
        # Normalize
        px_dir_x = dx / length
        px_dir_y = dy / length
        
        # Dot product with road direction
        # Note: road_dir is in meters space, px_dir is in pixel space
        # In pixel space, Y is inverted (increases downward), so we negate dy for road_dir
        dot = px_dir_x * rd_x + px_dir_y * (-rd_y)  # Flip Y for pixel coordinates
        
        if garden_type == "front":
            # Front garden should be toward the road (positive dot product)
            return dot > -0.3  # Allow some tolerance
        else:
            # Back garden should be away from the road (negative dot product)
            return dot < 0.3  # Allow some tolerance
    
    def _is_front_of_other_building(
        self,
        px: int,
        py: int,
        our_building_idx,
        garden_type: str
    ) -> bool:
        """
        Check if this pixel is in the FRONT garden zone of another building.
        
        If looking for a back garden pin, we should reject pixels that are
        actually in front of a neighboring building (their front garden).
        """
        if garden_type != "back":
            return False  # Only check for back gardens
        
        width, height = self.image_size
        
        for idx, bld in self.buildings.iterrows():
            if idx == our_building_idx:
                continue
            
            direction_info = self.building_directions.get(idx)
            if not direction_info:
                continue
            
            # Get this building's centroid in pixel coords
            other_cx, other_cy = direction_info.get("centroid_px", (None, None))
            if other_cx is None:
                c = bld.geometry.centroid
                other_cx = int((c.x - self.geo_bounds["west"]) / 
                              (self.geo_bounds["east"] - self.geo_bounds["west"]) * width)
                other_cy = int((self.geo_bounds["north"] - c.y) / 
                              (self.geo_bounds["north"] - self.geo_bounds["south"]) * height)
            
            # Distance from pixel to this building
            dist_to_other = np.sqrt((px - other_cx)**2 + (py - other_cy)**2)
            dist_to_other_m = dist_to_other * self.meters_per_pixel
            
            # Only check buildings within 15m
            if dist_to_other_m > 15:
                continue
            
            # Check if pixel is in FRONT of this building
            road_dir = direction_info.get("direction_to_road")
            if road_dir is None:
                continue
            
            try:
                rd_x = float(road_dir[0])
                rd_y = float(road_dir[1])
            except (IndexError, TypeError, ValueError):
                continue
            
            # Vector from other building to pixel
            dx = px - other_cx
            dy = py - other_cy
            length = np.sqrt(dx**2 + dy**2)
            
            if length < 1:
                continue
            
            px_dir_x = dx / length
            px_dir_y = dy / length
            
            dot = px_dir_x * rd_x + px_dir_y * (-rd_y)  # Flip Y for pixel coords
            
            # If pixel is in front of this other building (positive dot product)
            # and close to it, reject it - it's the neighbor's front garden
            if dot > 0.5 and dist_to_other_m < 12:
                return True  # This is in front of another building
        
        return False
    
    def _is_laterally_aligned(
        self,
        px: int,
        py: int,
        building_idx: int,
        garden_type: str,
        tolerance_m: float = 3.0
    ) -> bool:
        """
        Check if a pixel is laterally aligned with the building.
        
        The garden should be within the building's width corridor
        (perpendicular to road direction), not shifted sideways into
        a neighbor's space.
        
        We project the building's width perpendicular to the road direction,
        and check if the pixel falls within that corridor (plus tolerance).
        """
        if building_idx >= len(self.buildings):
            return True
        
        building = self.buildings.iloc[building_idx]
        building_idx_original = self.buildings.index[building_idx]
        direction_info = self.building_directions.get(building_idx_original)
        
        if not direction_info:
            return True  # No direction info, allow it
        
        road_dir = direction_info.get("direction_to_road")
        if road_dir is None:
            return True
        
        try:
            rd_x = float(road_dir[0])
            rd_y = float(road_dir[1])
        except (IndexError, TypeError, ValueError):
            return True
        
        # Get building bounds in pixel coords
        building_poly = self._geometry_to_pixel_polygon(building.geometry)
        if building_poly is None:
            return True
        
        # Get building centroid
        bld_cx, bld_cy = direction_info.get("centroid_px", (None, None))
        if bld_cx is None:
            bounds = building_poly.bounds
            bld_cx = (bounds[0] + bounds[2]) / 2
            bld_cy = (bounds[1] + bounds[3]) / 2
        
        # The "lateral" direction is perpendicular to the road direction
        # road_dir is (dx, dy) pointing toward road
        # lateral is (-dy, dx) or (dy, -dx)
        lateral_dir = (-rd_y, rd_x)  # Perpendicular in meters space
        lateral_dir_px = (lateral_dir[0], -lateral_dir[1])  # Flip Y for pixel space
        
        # Calculate the building's width along the lateral axis
        # by projecting building corners onto the lateral axis
        try:
            coords = list(building_poly.exterior.coords)
            lateral_positions = []
            for coord in coords:
                # Vector from centroid to corner
                dx = coord[0] - bld_cx
                dy = coord[1] - bld_cy
                # Project onto lateral axis
                proj = dx * lateral_dir_px[0] + dy * lateral_dir_px[1]
                lateral_positions.append(proj)
            
            min_lateral = min(lateral_positions)
            max_lateral = max(lateral_positions)
            
            # Add tolerance
            tolerance_px = tolerance_m / self.meters_per_pixel
            min_lateral -= tolerance_px
            max_lateral += tolerance_px
            
            # Project the pixel onto the lateral axis
            dx = px - bld_cx
            dy = py - bld_cy
            pixel_lateral = dx * lateral_dir_px[0] + dy * lateral_dir_px[1]
            
            # Check if pixel is within the building's lateral corridor
            return min_lateral <= pixel_lateral <= max_lateral
            
        except Exception:
            return True  # On error, allow it
    
    def _find_non_grass_pin(
        self,
        building_idx: int,
        bld_cx: int,
        bld_cy: int,
        zone_mask: np.ndarray,
        garden_type: str
    ) -> Optional[DeliveryPin]:
        """
        Fallback: find a driveway or paved area pin when no grass is available.
        """
        target_class = 1 if garden_type == "front" else 2
        height, width = self.classification_mask.shape
        
        search_radius_px = int(12 / self.meters_per_pixel)
        
        min_y = max(0, bld_cy - search_radius_px)
        max_y = min(height, bld_cy + search_radius_px)
        min_x = max(0, bld_cx - search_radius_px)
        max_x = min(width, bld_cx + search_radius_px)
        
        best_score = -1
        best_px, best_py = None, None
        best_surface = None
        
        for py in range(min_y, max_y, 2):
            for px in range(min_x, max_x, 2):
                if not zone_mask[py, px]:
                    continue
                if self.classification_mask[py, px] != target_class:
                    continue
                
                # Property border enforcement
                if hasattr(self, '_owner_edge_dist'):
                    edge_dist_m = self._owner_edge_dist[py, px] * self.meters_per_pixel
                    if edge_dist_m < config.PROPERTY_BORDER_WIDTH_M / 2:
                        continue
                
                # Lateral alignment: must be within building's width corridor
                if not self._is_laterally_aligned(
                    px, py, building_idx, garden_type
                ):
                    continue
                
                surface_type = self._get_surface_type(px, py)
                
                # Accept driveway, paved, or car parking
                if surface_type in (SurfaceType.BUILDING, SurfaceType.ROAD, SurfaceType.UNKNOWN):
                    continue
                
                score = self._calculate_score(px, py, surface_type)
                
                if score > best_score:
                    best_score = score
                    best_px, best_py = px, py
                    best_surface = surface_type
        
        if best_px is None:
            return None
        
        building = self.buildings.iloc[building_idx]
        lat, lon = self._pixel_to_geo(best_px, best_py)
        dist_m = self.distance_to_building[best_py, best_px] * self.meters_per_pixel
        
        building_id = None
        if "osm_id" in building.index:
            building_id = str(building["osm_id"])
        elif hasattr(building, 'name') and building.name is not None:
            building_id = str(building.name)
        
        return DeliveryPin(
            lat=lat,
            lon=lon,
            garden_type=garden_type,
            score=best_score,
            surface_type=best_surface.name.lower(),
            distance_to_building_m=dist_m,
            building_id=building_id,
            metadata={"pixel_x": best_px, "pixel_y": best_py}
        )


def find_delivery_pins_for_area(
    classification_mask: np.ndarray,
    vegetation_mask: np.ndarray,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    driveways: gpd.GeoDataFrame,
    geo_bounds: dict,
    image_size: Tuple[int, int],
    center_lat: float,
    center_lon: float,
    min_score: float = 20.0
) -> List[DeliveryPin]:
    """
    Convenience function to find all delivery pins for an area.
    
    Args:
        classification_mask: Classification mask
        vegetation_mask: Vegetation mask
        buildings: Buildings GeoDataFrame
        roads: Roads GeoDataFrame
        driveways: Driveways GeoDataFrame
        geo_bounds: Geographic bounds
        image_size: Image size
        center_lat: Center latitude
        center_lon: Center longitude
        min_score: Minimum score threshold
        
    Returns:
        List of DeliveryPin objects
    """
    finder = DeliveryPinFinder(
        classification_mask=classification_mask,
        vegetation_mask=vegetation_mask,
        buildings=buildings,
        roads=roads,
        driveways=driveways,
        geo_bounds=geo_bounds,
        image_size=image_size,
        center_lat=center_lat,
        center_lon=center_lon
    )
    
    return finder.find_all_pins(min_score=min_score)
