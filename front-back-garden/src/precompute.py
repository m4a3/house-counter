"""
Precomputation Module for Batch Processing

REWRITTEN for speed:
- Fetches OSM data ONCE for entire area (not per-chunk)
- Fetches tiles ONCE as a single stitched image
- Processes vegetation detection ONCE
- Classifies ONCE
- Then iterates buildings to find exactly ONE front + ONE back pin per building
- Deduplicates pins so no building gets multiple pins

Previous approach: 4 hours for 500m (25 chunks x separate fetch + process each)
New approach: ~1-3 minutes for 500m (1 fetch + 1 process + iterate buildings)
"""

import gc
import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
import threading
import warnings

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from tqdm import tqdm

import config
from src.tiles import (
    TileSource,
    fetch_area_image,
    recommended_zoom,
)
from src.osm import (
    fetch_buildings,
    fetch_roads,
    fetch_driveways,
    fetch_address_polygons,
    fetch_property_boundaries,
    geometry_to_pixel_coords,
    load_osm_from_cache,
    project_to_meters,
    save_osm_to_cache,
)
from src.garden_detector import detect_green_areas, detect_vegetation_enhanced, exclude_buildings_from_mask, exclude_roads_from_mask, split_vegetation_by_texture, recover_shaded_grass
from src.classifier import GardenClassifier
from src.delivery_pins import DeliveryPinFinder, DeliveryPin

# Precomputation cache directory
PRECOMPUTE_CACHE_DIR = Path(config.OUTPUT_DIR) / "precompute"


@dataclass
class PrecomputedArea:
    """Cached result for a precomputed area."""
    center_lat: float
    center_lon: float
    radius_m: float
    zoom: int
    tile_source: str
    computed_at: float
    num_buildings: int
    num_front_gardens: int
    num_back_gardens: int
    delivery_pins: List[Dict[str, Any]]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PrecomputedArea':
        return cls(**data)


class PrecomputeManager:
    """
    Manages precomputation of garden data for large areas.

    FAST single-pass approach:
    1. Fetch OSM buildings/roads/driveways ONCE for the whole area
    2. Fetch satellite imagery ONCE as a stitched image
    3. Detect vegetation ONCE
    4. Classify front/back ONCE
    5. Find exactly ONE front + ONE back pin per building
    """

    def __init__(
        self,
        cache_dir: Path = None,
        tile_source: TileSource = TileSource.AUTO,
        zoom: int = None,
        max_workers: int = 4
    ):
        self.cache_dir = cache_dir or PRECOMPUTE_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tile_source = tile_source
        self._preferred_zoom = zoom or config.ZOOM_LEVEL
        self.zoom = self._preferred_zoom  # May be overridden per-request
        self.max_workers = max_workers

        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, dict]:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        with self._lock:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)

    # ------------------------------------------------------------------
    # Cache key / path helpers
    # ------------------------------------------------------------------

    def _get_area_key(self, lat: float, lon: float, radius_m: float) -> str:
        key_data = f"{lat:.5f}_{lon:.5f}_{radius_m:.0f}_{self.zoom}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def _get_area_cache_path(self, area_key: str) -> Path:
        return self.cache_dir / f"area_{area_key}.pkl"

    # Backwards-compat: old chunk key for is_area_cached check
    def _get_chunk_key(self, lat: float, lon: float) -> str:
        return f"{round(lat, 4):.4f}_{round(lon, 4):.4f}_{self.zoom}"

    def _get_chunk_cache_path(self, chunk_key: str) -> Path:
        return self.cache_dir / f"chunk_{chunk_key}.pkl"

    # ------------------------------------------------------------------
    # Main precompute - SINGLE PASS
    # ------------------------------------------------------------------

    def precompute_area(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        parallel: bool = True,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Precompute all data for an area in a SINGLE PASS.

        Steps:
        1. Check cache
        2. Fetch imagery (one stitched image)
        3. Fetch OSM data (one set of API calls)
        4. Detect vegetation
        5. Classify front/back
        6. Find one pin per building per garden type
        7. Cache results
        """
        start_time = time.time()

        # Auto-select a safe zoom for this radius so the stitched image
        # stays within memory limits (~12k px per side → ~2-3 GB peak).
        safe_zoom = recommended_zoom(radius_m, center_lat, self._preferred_zoom)
        if safe_zoom != self._preferred_zoom and show_progress:
            print(f"⚠️  Zoom reduced {self._preferred_zoom} → {safe_zoom} "
                  f"for {radius_m:.0f}m radius (image would exceed memory limits)")
        self.zoom = safe_zoom

        area_key = self._get_area_key(center_lat, center_lon, radius_m)
        cache_path = self._get_area_cache_path(area_key)

        # Check cache
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if show_progress:
                    print(f"Loaded from cache: {len(cached.delivery_pins)} pins")
                elapsed = time.time() - start_time
                return {
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "radius_m": radius_m,
                    "total_pins": len(cached.delivery_pins),
                    "total_buildings": cached.num_buildings,
                    "elapsed_seconds": round(elapsed, 2),
                    "tile_source": self.tile_source.value,
                    "from_cache": True,
                }
            except Exception:
                pass

        # ---- STEP 1: Fetch imagery (ONCE) ----
        if show_progress:
            print(f"[1/5] Fetching satellite imagery for {radius_m}m radius...")
        t0 = time.time()

        image, metadata = fetch_area_image(
            center_lat, center_lon,
            radius_m=radius_m,
            zoom=self.zoom,
            show_progress=show_progress,
            use_cache=True,
            tile_source=self.tile_source,
        )

        if image is None:
            return {"error": "Failed to fetch imagery", "elapsed_seconds": time.time() - start_time}

        geo_bounds = metadata["geo_bounds"]
        image_size = tuple(metadata["image_size"])

        if show_progress:
            print(f"    Image: {image_size[0]}x{image_size[1]} ({time.time()-t0:.1f}s)")

        # ---- STEP 2: Fetch OSM data (ONCE, with cache) ----
        if show_progress:
            print(f"[2/5] Fetching OSM data...")
        t0 = time.time()

        osm_cached = load_osm_from_cache(center_lat, center_lon, radius_m)
        if osm_cached is not None:
            buildings = osm_cached["buildings"]
            roads = osm_cached["roads"]
            driveways = osm_cached["driveways"]
            address_polygons = osm_cached.get("address_polygons", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
            property_boundaries = osm_cached.get("property_boundaries", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
            if show_progress:
                print(f"    📦 OSM loaded from cache ({len(buildings)} buildings, "
                      f"{len(roads)} roads, {len(driveways)} driveways, "
                      f"{len(address_polygons)} addresses, "
                      f"{len(property_boundaries)} boundaries) ({time.time()-t0:.1f}s)")
        else:
            buildings = fetch_buildings(center_lat, center_lon, radius_m, show_progress=False)
            if show_progress:
                print(f"    {len(buildings)} buildings ({time.time()-t0:.1f}s)")
            
            t1 = time.time()
            roads = fetch_roads(center_lat, center_lon, radius_m, show_progress=False)
            if show_progress:
                print(f"    {len(roads)} roads ({time.time()-t1:.1f}s)")
            
            # Driveways, addresses, and boundaries are independent — fetch in parallel
            t1 = time.time()
            from concurrent.futures import ThreadPoolExecutor as _TPE
            _empty_gdf = lambda: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            def _fetch_driveways():
                return fetch_driveways(center_lat, center_lon, radius_m, show_progress=False)

            def _fetch_addresses():
                try:
                    return fetch_address_polygons(center_lat, center_lon, radius_m, show_progress=False)
                except Exception:
                    return _empty_gdf()

            def _fetch_boundaries():
                try:
                    return fetch_property_boundaries(center_lat, center_lon, radius_m, show_progress=False)
                except Exception:
                    return _empty_gdf()

            with _TPE(max_workers=3) as pool:
                fut_d = pool.submit(_fetch_driveways)
                fut_a = pool.submit(_fetch_addresses)
                fut_b = pool.submit(_fetch_boundaries)
                driveways = fut_d.result()
                address_polygons = fut_a.result()
                property_boundaries = fut_b.result()

            if show_progress:
                print(f"    {len(driveways)} driveways, {len(address_polygons)} addresses, "
                      f"{len(property_boundaries)} boundaries ({time.time()-t1:.1f}s)")
                print(f"    OSM total: {time.time()-t0:.1f}s")

            save_osm_to_cache({
                "buildings": buildings,
                "roads": roads,
                "driveways": driveways,
                "address_polygons": address_polygons,
                "property_boundaries": property_boundaries,
            }, center_lat, center_lon, radius_m)

        if buildings.empty:
            return {
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_m": radius_m,
                "total_pins": 0,
                "total_buildings": 0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "tile_source": self.tile_source.value,
                "from_cache": False,
            }

        # ---- STEP 3: Detect vegetation (ONCE) ----
        if show_progress:
            print(f"[3/5] Detecting vegetation...")
        t0 = time.time()

        # Two-stage vegetation detection:
        # 1. Broad HSV ranges (enhanced=True) to catch all possible grass
        # 2. Excess Green Index (ExG = 2G-R-B) confirmation to reject false positives
        # This prevents paved surfaces with slight green tint from being mis-classified,
        # while still catching shadowed/muted grass that a single HSV range would miss.
        # At lower zoom levels the ExG threshold is relaxed since colors are more muted.
        exg_threshold = 0.05 if self.zoom <= 17 else 0.08
        vegetation_mask = detect_vegetation_enhanced(image, use_texture=False)

        # Refine: re-apply ExG at our zoom-aware threshold (the function uses 0.1 internally)
        image_float = image.astype(np.float32) / 255.0
        exg = 2 * image_float[:,:,1] - image_float[:,:,0] - image_float[:,:,2]
        vegetation_mask[exg < exg_threshold] = 0
        del image_float, exg

        # Convert to pixel coords and exclude buildings/roads from vegetation
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

        vegetation_mask = exclude_buildings_from_mask(vegetation_mask, building_polys_px)
        vegetation_mask = exclude_roads_from_mask(vegetation_mask, road_lines_px, road_width_px=8)

        # Texture analysis: split vegetation into grass (smooth) vs tree canopy (rough)
        # Uses CV (std/mean) + building proximity recovery for robustness
        mpp = (geo_bounds["east"] - geo_bounds["west"]) * 111320 * np.cos(np.radians(center_lat)) / image_size[0]
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

        if show_progress:
            print(f"    Vegetation detected ({time.time()-t0:.1f}s)")

        # ---- STEP 4: Classify front/back (ONCE) ----
        if show_progress:
            print(f"[4/5] Classifying front/back gardens...")
        t0 = time.time()

        classifier = GardenClassifier(
            buildings=buildings,
            roads=roads,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=center_lat,
            center_lon=center_lon,
            driveways=driveways,
            exclusion_zones=None,        # Skip for speed
            property_boundaries=None,    # Skip for speed
            address_polygons=address_polygons,
        )

        # Create SPATIAL classification mask covering ALL potential garden areas
        # (not just grass). This ensures paved/concrete front gardens are also
        # classified as front/back, so the pin finder can place pins there.
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

        if show_progress:
            print(f"    Classification complete ({time.time()-t0:.1f}s)")

        gc.collect()

        # ---- STEP 5: Find pins - ONE per building per garden type ----
        if show_progress:
            print(f"[5/5] Finding delivery pins for {len(buildings)} buildings...")
        t0 = time.time()

        pin_finder = DeliveryPinFinder(
            classification_mask=classification_mask,
            vegetation_mask=grass_mask,  # Only grass, not trees
            buildings=buildings,
            roads=roads,
            driveways=driveways,
            geo_bounds=geo_bounds,
            image_size=image_size,
            center_lat=center_lat,
            center_lon=center_lon,
            building_directions=getattr(classifier, 'building_directions', None),
            property_boundaries=property_boundaries,
            tree_canopy_mask=tree_mask,  # For canopy density penalty
            original_vegetation_mask=vegetation_mask,  # Full veg mask before texture split
            building_polys_px=building_polys_px,  # Reuse from step 3
            road_lines_px=road_lines_px,           # Reuse from step 3
        )
        # Free large objects no longer needed -- pin_finder holds its own references.
        # This must happen BEFORE the pin loop triggers lazy inits that allocate more.
        # Keep building_directions for no_garden fallback pin placement.
        _building_directions = getattr(classifier, 'building_directions', {})
        del image, vegetation_mask, classification_mask, grass_mask, tree_mask
        del classifier, building_polys_px, road_lines_px
        gc.collect()

        all_pins: List[DeliveryPin] = []
        seen_buildings: Set[str] = set()

        for idx in tqdm(range(len(buildings)),
                        desc="Placing pins",
                        disable=not show_progress):
            # Get a stable building ID
            building = buildings.iloc[idx]
            building_id = None
            if "osm_id" in building.index:
                building_id = str(building["osm_id"])
            elif hasattr(building, 'name') and building.name is not None:
                building_id = str(building.name)
            else:
                building_id = f"bld_{idx}"

            # Skip if already processed (dedup)
            if building_id in seen_buildings:
                continue
            seen_buildings.add(building_id)

            # Find ONE front pin and ONE back pin for this building
            front_pin = pin_finder.find_best_pin_for_building(idx, "front")
            if front_pin:
                front_pin.building_id = building_id
                all_pins.append(front_pin)
            else:
                # No suitable front garden - place pin at EXPECTED garden location
                # (offset from building toward road, not at building centroid)
                centroid = building.geometry.centroid
                expected_lat, expected_lon = centroid.y, centroid.x
                bld_idx_original = buildings.index[idx]
                direction_info = _building_directions.get(bld_idx_original)
                if direction_info:
                    road_dir = direction_info.get("direction_to_road")
                    if road_dir is not None and np.linalg.norm(road_dir) > 0.1:
                        offset_m = 5.0
                        expected_lat += road_dir[1] * offset_m / 111320
                        expected_lon += road_dir[0] * offset_m / (111320 * np.cos(np.radians(center_lat)))
                all_pins.append(DeliveryPin(
                    lat=expected_lat, lon=expected_lon,
                    garden_type="front", score=0.0,
                    surface_type="no_garden",
                    distance_to_building_m=0.0,
                    building_id=building_id,
                    metadata={}
                ))

            back_pin = pin_finder.find_best_pin_for_building(idx, "back")
            if back_pin:
                back_pin.building_id = building_id
                all_pins.append(back_pin)
            else:
                # No suitable back garden - place pin at EXPECTED location (opposite road)
                centroid = building.geometry.centroid
                expected_lat, expected_lon = centroid.y, centroid.x
                bld_idx_original = buildings.index[idx]
                direction_info = _building_directions.get(bld_idx_original)
                if direction_info:
                    road_dir = direction_info.get("direction_to_road")
                    if road_dir is not None and np.linalg.norm(road_dir) > 0.1:
                        offset_m = 5.0
                        expected_lat -= road_dir[1] * offset_m / 111320
                        expected_lon -= road_dir[0] * offset_m / (111320 * np.cos(np.radians(center_lat)))
                all_pins.append(DeliveryPin(
                    lat=expected_lat, lon=expected_lon,
                    garden_type="back", score=0.0,
                    surface_type="no_garden",
                    distance_to_building_m=0.0,
                    building_id=building_id,
                    metadata={}
                ))

        # Pre-build building ID → positional index lookup (used by all
        # post-processing passes, avoids O(n) linear scans per building).
        bld_id_to_pos: Dict[str, int] = {}
        for pos_idx in range(len(buildings)):
            b = buildings.iloc[pos_idx]
            if "osm_id" in b.index:
                b_id = str(b["osm_id"])
            elif hasattr(b, "name") and b.name is not None:
                b_id = str(b.name)
            else:
                b_id = f"bld_{pos_idx}"
            bld_id_to_pos[b_id] = pos_idx

        # ---- POST-PROCESSING: Validate front/back pin placement ----
        # Rule: The front pin MUST be closer to a road than the back pin.
        # If not, swap them.
        # Reuse pin_finder's road distance transform (already computed).
        _road_dist_val = pin_finder.distance_to_road
        _h, _w = image_size[1], image_size[0]

        swapped_count = 0
        pins_by_bld = {}
        for p in all_pins:
            bid = p.building_id
            if bid not in pins_by_bld:
                pins_by_bld[bid] = {}
            pins_by_bld[bid][p.garden_type] = p

        for bid, bpins in pins_by_bld.items():
            fp = bpins.get("front")
            bp = bpins.get("back")
            if fp is None or bp is None:
                continue
            if fp.score == 0 or bp.score == 0:
                continue

            # Convert to pixels
            fpx = int((fp.lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * _w)
            fpy = int((geo_bounds["north"] - fp.lat) / (geo_bounds["north"] - geo_bounds["south"]) * _h)
            bpx = int((bp.lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * _w)
            bpy = int((geo_bounds["north"] - bp.lat) / (geo_bounds["north"] - geo_bounds["south"]) * _h)

            fpx = max(0, min(_w - 1, fpx))
            fpy = max(0, min(_h - 1, fpy))
            bpx = max(0, min(_w - 1, bpx))
            bpy = max(0, min(_h - 1, bpy))

            front_road_dist = float(_road_dist_val[fpy, fpx])
            back_road_dist = float(_road_dist_val[bpy, bpx])
            
            # Check separation between front and back pins
            sep_m = np.sqrt(((fp.lat - bp.lat) * 111320)**2 +
                            ((fp.lon - bp.lon) * 111320 * np.cos(np.radians(center_lat)))**2)
            
            should_swap = False
            
            # Rule 1: Back pin is closer to road than front → swap
            if front_road_dist > back_road_dist * 1.05:
                should_swap = True
            
            # Rule 2: Both pins within 8m of each other (same garden) and
            # back pin is at least as close to road → swap
            if sep_m < 8 and back_road_dist <= front_road_dist:
                should_swap = True
            
            if should_swap:
                fp.garden_type = "back"
                bp.garden_type = "front"
                swapped_count += 1

        del _road_dist_val

        if show_progress and swapped_count > 0:
            print(f"    Front/back swap correction: {swapped_count} buildings fixed")

        # ---- POST-PROCESSING: Same-side pin correction ----
        # When front and back pins are very close together, they're in the
        # same garden.  Re-search for the lower-scored pin on the opposite
        # side of the building, then assign front/back labels by road distance
        # (front = closer to road).
        _road_dist = pin_finder.distance_to_road
        _h2, _w2 = image_size[1], image_size[0]
        same_side_fixed = 0
        same_side_label_swapped = 0
        for bid, bpins in pins_by_bld.items():
            fp = bpins.get("front")
            bp = bpins.get("back")
            if fp is None or bp is None:
                continue
            if fp.score == 0 or bp.score == 0:
                continue

            sep_m = np.sqrt(
                ((fp.lat - bp.lat) * 111320) ** 2
                + ((fp.lon - bp.lon) * 111320 * np.cos(np.radians(center_lat))) ** 2
            )
            if sep_m >= 8:
                continue

            # Determine which pin to keep (higher score stays)
            keep, redo = (fp, bp) if fp.score >= bp.score else (bp, fp)
            redo_type = redo.garden_type

            # Direction from building centroid to kept pin (in pixel space)
            keep_px = int(
                (keep.lon - geo_bounds["west"])
                / (geo_bounds["east"] - geo_bounds["west"])
                * _w2
            )
            keep_py = int(
                (geo_bounds["north"] - keep.lat)
                / (geo_bounds["north"] - geo_bounds["south"])
                * _h2
            )

            bld_pos_idx = bld_id_to_pos.get(bid)
            if bld_pos_idx is None:
                continue

            # Building centroid in pixels
            c = buildings.iloc[bld_pos_idx].geometry.centroid
            bld_cx = int(
                (c.x - geo_bounds["west"])
                / (geo_bounds["east"] - geo_bounds["west"])
                * _w2
            )
            bld_cy = int(
                (geo_bounds["north"] - c.y)
                / (geo_bounds["north"] - geo_bounds["south"])
                * _h2
            )

            exclude_dir = (keep_px - bld_cx, keep_py - bld_cy)
            if abs(exclude_dir[0]) < 1 and abs(exclude_dir[1]) < 1:
                continue

            new_pin = pin_finder.find_best_pin_for_building(
                bld_pos_idx, redo_type, exclude_direction=exclude_dir
            )
            if new_pin and new_pin.score > 0:
                redo.lat = new_pin.lat
                redo.lon = new_pin.lon
                redo.score = new_pin.score
                redo.surface_type = new_pin.surface_type
                redo.distance_to_building_m = new_pin.distance_to_building_m
                redo.metadata = new_pin.metadata
                same_side_fixed += 1
            else:
                offset_m = 5.0
                dlat = c.y - keep.lat
                dlon = c.x - keep.lon
                geo_len = np.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * np.cos(np.radians(center_lat))) ** 2)
                if geo_len > 0.5:
                    opp_lat = c.y + dlat / geo_len * offset_m / 111320
                    opp_lon = c.x + dlon / geo_len * offset_m / (111320 * np.cos(np.radians(center_lat)))
                else:
                    opp_lat, opp_lon = c.y, c.x
                redo.lat = opp_lat
                redo.lon = opp_lon
                redo.score = 0.0
                redo.surface_type = "no_garden"
                redo.distance_to_building_m = 0.0
                redo.metadata = {}
                same_side_fixed += 1

            # After relocation, assign front/back labels by road distance:
            # the pin closer to a road is "front", the other is "back".
            # Use garden_type to find current front/back (dict keys may
            # disagree after earlier swaps).
            if fp.garden_type == "front":
                cur_fp, cur_bp = fp, bp
            else:
                cur_fp, cur_bp = bp, fp

            cfpx = max(0, min(_w2 - 1, int((cur_fp.lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * _w2)))
            cfpy = max(0, min(_h2 - 1, int((geo_bounds["north"] - cur_fp.lat) / (geo_bounds["north"] - geo_bounds["south"]) * _h2)))
            cbpx = max(0, min(_w2 - 1, int((cur_bp.lon - geo_bounds["west"]) / (geo_bounds["east"] - geo_bounds["west"]) * _w2)))
            cbpy = max(0, min(_h2 - 1, int((geo_bounds["north"] - cur_bp.lat) / (geo_bounds["north"] - geo_bounds["south"]) * _h2)))

            f_road = float(_road_dist[cfpy, cfpx])
            b_road = float(_road_dist[cbpy, cbpx])

            if f_road > b_road * 1.05:
                cur_fp.garden_type = "back"
                cur_bp.garden_type = "front"
                same_side_label_swapped += 1

        del _road_dist

        if show_progress and same_side_fixed > 0:
            print(f"    Same-side pin correction: {same_side_fixed} buildings relocated, {same_side_label_swapped} labels swapped")

        # ---- POST-PROCESSING: Address-road distance label fix ----
        # For buildings matched to a named street, measure each pin's
        # distance to THAT specific road (not any road).  The pin closer
        # to the address-matched road is "front".  This avoids false
        # swaps from walkways/back lanes in the generic road raster.
        has_addrs = (
            address_polygons is not None
            and not address_polygons.empty
            and "addr:street" in address_polygons.columns
        )
        addr_road_fixed = 0
        addr_road_fixed_bids: Set[str] = set()
        if has_addrs:
            from shapely.ops import nearest_points as _np_addr

            roads_m_pp = project_to_meters(roads, center_lat, center_lon)
            buildings_m_pp = project_to_meters(buildings, center_lat, center_lon)
            addrs_m_pp = project_to_meters(address_polygons, center_lat, center_lon)

            street_road_geoms: Dict[str, List] = {}
            if "name" in roads.columns:
                for ri in range(len(roads)):
                    rname = roads.iloc[ri].get("name", None)
                    if isinstance(rname, str) and len(rname) > 1:
                        key = rname.strip().lower()
                        if key not in street_road_geoms:
                            street_road_geoms[key] = []
                        street_road_geoms[key].append(roads_m_pp.iloc[ri].geometry)

            for bid, bpins in pins_by_bld.items():
                pin_a = bpins.get("front")
                pin_b = bpins.get("back")
                if pin_a is None or pin_b is None:
                    continue
                if pin_a.garden_type == "front":
                    cur_fp, cur_bp = pin_a, pin_b
                else:
                    cur_fp, cur_bp = pin_b, pin_a
                if cur_fp.score == 0 or cur_bp.score == 0:
                    continue

                pos_idx = bld_id_to_pos.get(bid)
                if pos_idx is None:
                    continue
                bld_idx_orig = buildings.index[pos_idx]
                dir_info = _building_directions.get(bld_idx_orig)
                if dir_info is None or dir_info.get("source") not in ("address", "driveway"):
                    continue

                bld_geom_m = buildings_m_pp.iloc[pos_idx].geometry
                nearby_addrs = list(addrs_m_pp.sindex.query(bld_geom_m.buffer(5), predicate=None))
                matched_street = None
                for ap in nearby_addrs:
                    aw = address_polygons.iloc[ap]
                    sn = aw.get("addr:street", None)
                    if isinstance(sn, str) and sn.strip().lower() in street_road_geoms:
                        matched_street = sn.strip().lower()
                        break
                if matched_street is None:
                    continue

                seg_list = street_road_geoms[matched_street]

                # Distance from each pin to the matched street (in metres)
                bld_cent = buildings_m_pp.iloc[pos_idx].geometry.centroid
                cos_r = np.cos(np.radians(center_lat))
                def _pin_m(plat, plon):
                    """Approximate metres offset from projected building centroid."""
                    dx = (plon - buildings.iloc[pos_idx].geometry.centroid.x) * 111320 * cos_r
                    dy = (plat - buildings.iloc[pos_idx].geometry.centroid.y) * 111320
                    return Point(bld_cent.x + dx, bld_cent.y + dy)

                fp_m = _pin_m(cur_fp.lat, cur_fp.lon)
                bp_m = _pin_m(cur_bp.lat, cur_bp.lon)

                # Each pin finds its own nearest segment of the address
                # road.  The pin closer to any segment is "front".
                fd = min(fp_m.distance(seg) for seg in seg_list)
                bd = min(bp_m.distance(seg) for seg in seg_list)

                if fd > bd * 1.05:
                    cur_fp.garden_type = "back"
                    cur_bp.garden_type = "front"
                    addr_road_fixed += 1
                    addr_road_fixed_bids.add(bid)

            del roads_m_pp, buildings_m_pp, addrs_m_pp

        if show_progress and addr_road_fixed > 0:
            print(f"    Address-road distance fix: {addr_road_fixed} buildings corrected")

        # ---- POST-PROCESSING: Neighbor consistency (conservative) ----
        # Only fix buildings that weren't already corrected by the
        # address-road fix.  Requires unanimous neighbor disagreement.
        # Uses a cKDTree for O(n log n) neighbor lookups instead of O(n^2).
        from scipy.spatial import cKDTree as _cKDTree

        cos_lat = np.cos(np.radians(center_lat))
        nc_bids: list[str] = []
        nc_dirs: list[tuple[float, float]] = []
        nc_coords_m: list[tuple[float, float]] = []
        for bid, bpins in pins_by_bld.items():
            pa, pb = bpins.get("front"), bpins.get("back")
            if pa is None or pb is None:
                continue
            fp_nc = pa if pa.garden_type == "front" else pb
            bp_nc = pb if pa.garden_type == "front" else pa
            if fp_nc.score == 0 or bp_nc.score == 0:
                continue
            dx = (fp_nc.lon - bp_nc.lon) * cos_lat
            dy = fp_nc.lat - bp_nc.lat
            length = np.sqrt(dx**2 + dy**2)
            if length > 1e-9:
                nc_bids.append(bid)
                nc_dirs.append((dx / length, dy / length))
                mid_lat = (fp_nc.lat + bp_nc.lat) / 2
                mid_lon = (fp_nc.lon + bp_nc.lon) / 2
                nc_coords_m.append((mid_lat * 111320, mid_lon * 111320 * cos_lat))

        neighbor_fixed = 0
        if nc_coords_m:
            tree = _cKDTree(np.array(nc_coords_m))
            neighbor_lists = tree.query_ball_tree(tree, r=20.0)

            for i, neighbors in enumerate(neighbor_lists):
                bid = nc_bids[i]
                if bid in addr_road_fixed_bids:
                    continue
                d = nc_dirs[i]
                agree = 0
                disagree = 0
                for j in neighbors:
                    if j == i:
                        continue
                    nd = nc_dirs[j]
                    dot = d[0] * nd[0] + d[1] * nd[1]
                    if dot > 0:
                        agree += 1
                    else:
                        disagree += 1

                if disagree >= 2 and agree == 0:
                    pa = pins_by_bld[bid].get("front")
                    pb = pins_by_bld[bid].get("back")
                    if pa is not None and pb is not None:
                        if pa.garden_type == "front":
                            pa.garden_type = "back"
                            pb.garden_type = "front"
                        else:
                            pa.garden_type = "front"
                            pb.garden_type = "back"
                        nc_dirs[i] = (-d[0], -d[1])
                        neighbor_fixed += 1

        if show_progress and neighbor_fixed > 0:
            print(f"    Neighbor consistency fix: {neighbor_fixed} buildings corrected")

        if show_progress:
            # Report attempt distribution
            from collections import Counter
            attempt_counts = Counter()
            for p in all_pins:
                att = p.metadata.get("attempt", 0)
                attempt_counts[att] += 1
            total_with_attempt = sum(v for k, v in attempt_counts.items() if k > 0)
            if total_with_attempt > 0:
                parts = []
                for att in sorted(attempt_counts.keys()):
                    if att == 0:
                        continue
                    count = attempt_counts[att]
                    pct = 100 * count / total_with_attempt
                    parts.append(f"A{att}={count}({pct:.0f}%)")
                print(f"    Attempt stats: {', '.join(parts)}, no_garden={attempt_counts.get(0, 0)}")
            print(f"    Found {len(all_pins)} pins ({time.time()-t0:.1f}s)")

        # Save count before freeing references
        num_buildings = len(buildings)

        # Free remaining large objects before caching
        del pin_finder, _building_directions
        del buildings, roads, driveways, address_polygons, property_boundaries
        gc.collect()

        # ---- Cache results ----
        num_front = sum(1 for p in all_pins if p.garden_type == "front" and p.score > 0)
        num_back = sum(1 for p in all_pins if p.garden_type == "back" and p.score > 0)
        num_no_garden = sum(1 for p in all_pins if p.score == 0)

        precomputed = PrecomputedArea(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_m=radius_m,
            zoom=self.zoom,
            tile_source=self.tile_source.value if hasattr(self.tile_source, 'value') else str(self.tile_source),
            computed_at=time.time(),
            num_buildings=num_buildings,
            num_front_gardens=num_front,
            num_back_gardens=num_back,
            delivery_pins=[pin.to_dict() for pin in all_pins],
        )

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(precomputed, f)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}")

        # Update index
        with self._lock:
            self.index[area_key] = {
                "lat": center_lat,
                "lon": center_lon,
                "radius_m": radius_m,
                "computed_at": precomputed.computed_at,
                "num_pins": len(all_pins),
                "num_buildings": num_buildings,
            }
        self._save_index()

        elapsed = time.time() - start_time

        if show_progress:
            print(f"\nPrecompute complete:")
            print(f"   Buildings: {num_buildings}")
            print(f"   Front pins: {num_front}")
            print(f"   Back pins: {num_back}")
            print(f"   No garden: {num_no_garden}")
            print(f"   Total time: {elapsed:.1f}s")

        return {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_m": radius_m,
            "total_pins": len(all_pins),
            "total_buildings": num_buildings,
            "num_front": num_front,
            "num_back": num_back,
            "num_no_garden": num_no_garden,
            "elapsed_seconds": round(elapsed, 2),
            "tile_source": self.tile_source.value if hasattr(self.tile_source, 'value') else str(self.tile_source),
            "from_cache": False,
        }

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def is_area_cached(self, center_lat: float, center_lon: float, radius_m: float) -> bool:
        """Check if the exact area is cached (accounts for zoom auto-reduction)."""
        # Check with current zoom
        area_key = self._get_area_key(center_lat, center_lon, radius_m)
        if self._get_area_cache_path(area_key).exists():
            return True
        # Also check with the zoom that would actually be used after auto-reduction
        safe_zoom = recommended_zoom(radius_m, center_lat, self._preferred_zoom)
        if safe_zoom != self.zoom:
            saved = self.zoom
            self.zoom = safe_zoom
            area_key = self._get_area_key(center_lat, center_lon, radius_m)
            self.zoom = saved
            return self._get_area_cache_path(area_key).exists()
        return False

    def get_pins_in_radius(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float,
        garden_type: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get all delivery pins within a radius.

        Tries exact cache first. If not found, checks for any cached area
        that covers the requested radius. Falls back to computing on-the-fly.
        """
        # Try exact cache
        area_key = self._get_area_key(center_lat, center_lon, radius_m)
        cache_path = self._get_area_cache_path(area_key)

        precomputed = None

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    precomputed = pickle.load(f)
            except Exception:
                pass

        # Try to find a larger cached area that covers this request
        if precomputed is None:
            precomputed = self._find_covering_cache(center_lat, center_lon, radius_m)

        # Compute on-the-fly if no cache
        if precomputed is None:
            summary = self.precompute_area(center_lat, center_lon, radius_m, show_progress=True)
            if "error" in summary:
                return []
            # Re-compute cache path: precompute_area may have changed self.zoom
            # (e.g. auto-reduced from 19 to 17 for large radii)
            area_key = self._get_area_key(center_lat, center_lon, radius_m)
            cache_path = self._get_area_cache_path(area_key)
            # Re-load from cache
            try:
                with open(cache_path, "rb") as f:
                    precomputed = pickle.load(f)
            except Exception:
                return []

        # Filter pins by distance, type, and score
        pins = []
        for pin in precomputed.delivery_pins:
            dist = self._haversine_distance(center_lat, center_lon, pin["lat"], pin["lon"])
            if dist > radius_m:
                continue
            if garden_type and pin.get("garden_type") != garden_type:
                continue
            if pin.get("score", 0) < min_score:
                continue
            pins.append(pin)

        return pins

    def _find_covering_cache(self, lat: float, lon: float, radius_m: float) -> Optional[PrecomputedArea]:
        """Find a cached area that fully covers the requested area."""
        for key, info in self.index.items():
            cached_lat = info.get("lat", 0)
            cached_lon = info.get("lon", 0)
            cached_radius = info.get("radius_m", 0)

            # Check if the cached area fully contains our requested area
            dist_between_centers = self._haversine_distance(lat, lon, cached_lat, cached_lon)
            if dist_between_centers + radius_m <= cached_radius:
                cache_path = self._get_area_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            return pickle.load(f)
                    except Exception:
                        continue
        return None

    def get_nearest_building_pins(
        self,
        lat: float,
        lon: float
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get front and back garden pins for the building nearest to a coordinate.

        Searches ALL cached areas (regardless of zoom level) that contain
        the point.  When multiple areas match, prefers the smallest area
        (highest resolution).  Groups pins by building_id and returns the
        paired front+back from the single nearest building, so both pins
        always belong to the same property.
        """
        # Collect every cached area that contains the query point,
        # sorted by radius ascending so we try the highest-resolution first.
        matching: list[tuple[float, str, dict]] = []
        for key, info in self.index.items():
            cached_lat = info.get("lat", 0)
            cached_lon = info.get("lon", 0)
            cached_radius = info.get("radius_m", 0)

            dist = self._haversine_distance(lat, lon, cached_lat, cached_lon)
            if dist <= cached_radius:
                matching.append((cached_radius, key, info))

        if not matching:
            print(f"    Precompute cache: ({lat:.5f}, {lon:.5f}) not inside any cached area")
            return {"front": None, "back": None}

        matching.sort(key=lambda t: t[0])

        for cached_radius, key, info in matching:
            cache_path = self._get_area_cache_path(key)
            if not cache_path.exists():
                continue
            try:
                with open(cache_path, "rb") as f:
                    precomputed = pickle.load(f)

                # Group pins by building
                buildings_map: Dict[str, Dict[str, Any]] = {}
                for pin in precomputed.delivery_pins:
                    bid = pin.get("building_id")
                    if bid is None:
                        continue
                    if bid not in buildings_map:
                        buildings_map[bid] = {}
                    buildings_map[bid][pin["garden_type"]] = pin

                # Find the building whose centroid is closest
                best_bid = None
                best_dist = float("inf")
                for bid, bpins in buildings_map.items():
                    lats = [p["lat"] for p in bpins.values()]
                    lons = [p["lon"] for p in bpins.values()]
                    clat = sum(lats) / len(lats)
                    clon = sum(lons) / len(lons)
                    d = self._haversine_distance(lat, lon, clat, clon)
                    if d < best_dist:
                        best_dist = d
                        best_bid = bid

                if best_bid is not None:
                    bpins = buildings_map[best_bid]
                    print(f"    Precompute cache hit: ({lat:.5f}, {lon:.5f}) → "
                          f"area r={cached_radius:.0f}m, zoom={getattr(precomputed, 'zoom', '?')}, "
                          f"nearest building {best_bid} ({best_dist:.1f}m away)")
                    return {
                        "front": bpins.get("front"),
                        "back": bpins.get("back"),
                    }
            except Exception:
                continue

        print(f"    Precompute cache: ({lat:.5f}, {lon:.5f}) inside {len(matching)} area(s) "
              f"but no building pins found")
        return {"front": None, "back": None}

    def classify_point(self, lat: float, lon: float) -> Dict[str, Any]:
        """Classify a single GPS coordinate using cached data if available."""
        # Check if any cached area covers this point
        for key, info in self.index.items():
            cached_lat = info.get("lat", 0)
            cached_lon = info.get("lon", 0)
            cached_radius = info.get("radius_m", 0)

            dist = self._haversine_distance(lat, lon, cached_lat, cached_lon)
            if dist <= cached_radius:
                cache_path = self._get_area_cache_path(key)
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            precomputed = pickle.load(f)

                        # Find nearest pin to classify
                        nearest_pin = None
                        min_dist = float("inf")
                        for pin in precomputed.delivery_pins:
                            d = self._haversine_distance(lat, lon, pin["lat"], pin["lon"])
                            if d < min_dist:
                                min_dist = d
                                nearest_pin = pin

                        if nearest_pin and min_dist < 30:
                            return {
                                "lat": lat,
                                "lon": lon,
                                "classification": f"{nearest_pin['garden_type']}_garden",
                                "surface_type": nearest_pin.get("surface_type", "unknown"),
                                "score": nearest_pin.get("score", 0),
                                "distance_to_building_m": nearest_pin.get("distance_to_building_m", 0),
                            }
                    except Exception:
                        continue

        return {"lat": lat, "lon": lon, "classification": "unknown", "score": 0}

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def clear_cache(self):
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()
        print("Precompute cache cleared.")

    def get_cache_stats(self) -> Dict[str, Any]:
        total_chunks = len(self.index)
        total_pins = sum(entry.get("num_pins", 0) for entry in self.index.values())
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl")) if self.cache_dir.exists() else 0
        return {
            "total_areas": total_chunks,
            "total_pins": total_pins,
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "cached_areas": [
                {"key": k, "lat": v.get("lat"), "lon": v.get("lon"), "radius_m": v.get("radius_m"), "pins": v.get("num_pins", 0)}
                for k, v in self.index.items()
            ],
        }


def precompute_area(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    tile_source: str = "auto",
    zoom: int = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """Convenience function to precompute an area."""
    source_map = {
        "google": TileSource.GOOGLE,
        "manna": TileSource.MANNA,
        "auto": TileSource.AUTO,
    }
    manager = PrecomputeManager(
        tile_source=source_map.get(tile_source, TileSource.AUTO),
        zoom=zoom,
    )
    return manager.precompute_area(
        center_lat=center_lat,
        center_lon=center_lon,
        radius_m=radius_m,
        parallel=parallel,
    )
