"""
Building data disk cache manager.

Caches Overture Maps building footprints as local GeoParquet files
for fast repeated queries without S3 round-trips.
"""

import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import geopandas as gpd
import pyarrow as pa

import overturemaps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path("building_cache")
AVG_BYTES_PER_BUILDING = 160  # empirical avg after dropping nested columns
DEFAULT_DENSITY_PER_KM2 = 200  # rough avg; cities include lots of open land
MAX_CACHE_AREA_KM2 = 10000  # ~100 km × 100 km safety limit

# Columns with deeply-nested structs that bloat files and may fail round-trip
_DROP_COLUMNS = frozenset({
    "bbox", "sources", "names", "categories", "brand",
    "addresses", "websites", "socials", "emails", "phones",
})


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_default_instance: Optional["CacheManager"] = None


def get_cache_manager() -> "CacheManager":
    """Return the process-wide CacheManager singleton."""
    global _default_instance
    if _default_instance is None:
        _default_instance = CacheManager()
    return _default_instance


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------
class CacheManager:
    """Manages a directory of cached GeoParquet files."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index_mtime: float = 0
        self._index: dict = self._load_index()

    # -- Index persistence -----------------------------------------------------

    def _index_path(self) -> Path:
        return self.cache_dir / "cache_index.json"

    def _load_index(self) -> dict:
        p = self._index_path()
        if p.exists():
            self._index_mtime = p.stat().st_mtime
            with open(p) as f:
                return json.load(f)
        return {"areas": []}

    def _save_index(self):
        p = self._index_path()
        with open(p, "w") as f:
            json.dump(self._index, f, indent=2)
        self._index_mtime = p.stat().st_mtime

    def _ensure_fresh(self):
        """Reload the index when the file has been modified externally."""
        p = self._index_path()
        if p.exists() and p.stat().st_mtime > self._index_mtime:
            self._index = self._load_index()

    # -- Read operations -------------------------------------------------------

    def get_cached_areas(self) -> List[dict]:
        with self._lock:
            self._ensure_fresh()
            return list(self._index["areas"])

    def get_stats(self) -> dict:
        areas = self.get_cached_areas()
        total_bytes = sum(a.get("file_size_bytes", 0) for a in areas)
        total_bldgs = sum(a.get("building_count", 0) for a in areas)
        return {
            "total_areas": len(areas),
            "total_buildings": total_bldgs,
            "total_size_bytes": total_bytes,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2) if total_bytes else 0,
        }

    def estimate_cache_size(self, bbox: Tuple[float, float, float, float]) -> dict:
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2
        lat_km = (max_lat - min_lat) * 111.32
        lon_km = (max_lon - min_lon) * 111.32 * math.cos(math.radians(center_lat))
        area_km2 = abs(lat_km * lon_km)
        est_buildings = int(area_km2 * DEFAULT_DENSITY_PER_KM2)
        est_bytes = est_buildings * AVG_BYTES_PER_BUILDING
        return {
            "bbox": list(bbox),
            "area_km2": round(area_km2, 2),
            "estimated_buildings": est_buildings,
            "estimated_size_bytes": est_bytes,
            "estimated_size_mb": round(est_bytes / (1024 * 1024), 2),
        }

    def find_covering_cache(
        self, bbox: Tuple[float, float, float, float]
    ) -> Optional[dict]:
        """Return a cached area that fully covers *bbox*, or None."""
        min_lon, min_lat, max_lon, max_lat = bbox
        with self._lock:
            self._ensure_fresh()
            areas = list(self._index["areas"])
        for a in areas:
            ab = a["bbox"]
            if (ab[0] <= min_lon and ab[1] <= min_lat
                    and ab[2] >= max_lon and ab[3] >= max_lat):
                return a
        return None

    def find_overlapping(
        self, bbox: Tuple[float, float, float, float]
    ) -> List[dict]:
        """Return cached areas that overlap with *bbox*."""
        min_lon, min_lat, max_lon, max_lat = bbox
        result = []
        with self._lock:
            self._ensure_fresh()
            areas = list(self._index["areas"])
        for a in areas:
            ab = a["bbox"]
            if not (ab[2] < min_lon or ab[0] > max_lon
                    or ab[3] < min_lat or ab[1] > max_lat):
                result.append(a)
        return result

    def load_geodataframe(self, area_id: str) -> Optional[gpd.GeoDataFrame]:
        """Load a cached area as a GeoDataFrame."""
        with self._lock:
            self._ensure_fresh()
            target = None
            for a in self._index["areas"]:
                if a["id"] == area_id:
                    target = a
                    break
        if target is None:
            return None
        fp = self.cache_dir / target["file_path"]
        if not fp.exists():
            return None
        gdf = gpd.read_parquet(str(fp))
        # Update last-accessed timestamp
        with self._lock:
            target["last_accessed"] = datetime.now(timezone.utc).isoformat()
            self._save_index()
        return gdf

    # -- Write operations ------------------------------------------------------

    def cache_area(
        self,
        bbox: Tuple[float, float, float, float],
        name: str,
        center_lat: float,
        center_lon: float,
        radius_km: Optional[float] = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
    ) -> Optional[dict]:
        """Download building data for *bbox* and persist as GeoParquet.

        Returns the area metadata dict on success, or None if no buildings
        were found.
        """

        def _progress(pct: int, msg: str):
            if progress_cb:
                progress_cb(pct, msg)

        # Validate area size
        est = self.estimate_cache_size(bbox)
        if est["area_km2"] > MAX_CACHE_AREA_KM2:
            raise ValueError(
                f"Area too large ({est['area_km2']:.0f} km²). "
                f"Maximum is {MAX_CACHE_AREA_KM2} km²."
            )

        area_id = uuid.uuid4().hex[:8]
        file_name = f"{area_id}.parquet"
        full_path = self.cache_dir / file_name

        _progress(2, "Connecting to Overture Maps…")

        # -- Stream record batches for progress reporting ----------------------
        reader = overturemaps.record_batch_reader("building", bbox)
        schema: Optional[pa.Schema] = None
        batches: list[pa.RecordBatch] = []
        total_rows = 0
        expected = max(est["estimated_buildings"], 500)

        _progress(5, "Downloading building data…")

        for batch in reader:
            if schema is None:
                schema = batch.schema
            batches.append(batch)
            total_rows += batch.num_rows
            pct = min(5 + int(65 * total_rows / expected), 70)
            _progress(pct, f"Downloaded {total_rows:,} buildings…")

        if not batches or total_rows == 0:
            _progress(100, "No buildings found in this area.")
            return None

        _progress(72, f"Merging {total_rows:,} records…")
        assert schema is not None
        table = pa.Table.from_batches(batches, schema=schema)

        # Drop problematic nested columns
        keep_cols = [c for c in table.column_names if c not in _DROP_COLUMNS]
        table = table.select(keep_cols)

        _progress(75, "Converting geometry…")
        df = table.to_pandas()

        if "geometry" not in df.columns:
            _progress(100, "No geometry column — cannot cache.")
            return None

        from shapely import wkb as _wkb

        if len(df) > 0 and isinstance(df["geometry"].iloc[0], bytes):
            df["geometry"] = df["geometry"].apply(
                lambda g: _wkb.loads(g) if g else None
            )

        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        _progress(85, "Saving to disk…")
        gdf.to_parquet(str(full_path))
        file_size = full_path.stat().st_size

        # Compute area
        min_lon, min_lat, max_lon, max_lat = bbox
        c_lat = (min_lat + max_lat) / 2
        lat_km = (max_lat - min_lat) * 111.32
        lon_km = (max_lon - min_lon) * 111.32 * math.cos(math.radians(c_lat))
        area_km2 = abs(lat_km * lon_km)

        now = datetime.now(timezone.utc).isoformat()
        area_entry = {
            "id": area_id,
            "name": name,
            "bbox": list(bbox),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "radius_km": radius_km,
            "area_km2": round(area_km2, 2),
            "building_count": len(gdf),
            "file_size_bytes": file_size,
            "file_path": file_name,
            "created_at": now,
            "last_accessed": now,
        }

        with self._lock:
            self._index["areas"].append(area_entry)
            self._save_index()

        _progress(
            100,
            f"Cached {len(gdf):,} buildings "
            f"({file_size / (1024 * 1024):.1f} MB)",
        )
        return area_entry

    def delete_area(self, area_id: str) -> bool:
        with self._lock:
            for i, a in enumerate(self._index["areas"]):
                if a["id"] == area_id:
                    fp = self.cache_dir / a["file_path"]
                    if fp.exists():
                        fp.unlink()
                    self._index["areas"].pop(i)
                    self._save_index()
                    return True
        return False

    def clear_all(self) -> int:
        """Delete every cached area. Returns count of areas removed."""
        with self._lock:
            count = len(self._index["areas"])
            for a in self._index["areas"]:
                fp = self.cache_dir / a["file_path"]
                if fp.exists():
                    fp.unlink()
            self._index["areas"] = []
            self._save_index()
        return count
