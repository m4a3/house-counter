"""
OSM Contributor store + Overture-only candidate detection.

Each saved contribution is one GeoJSON ``Feature`` stored as
``contributions/<uuid>.json``. The folder is the source of truth;
listing is done by globbing the directory. Concurrent writes are
safe because each contribution gets a unique UUID filename; an
in-process ``threading.Lock`` guards the listing / index reads to
match the pattern used by ``cache_manager.CacheManager``.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pyproj
from shapely.geometry import Polygon, mapping, shape
from shapely.geometry.polygon import orient as _shapely_orient
from shapely.ops import transform
from shapely.validation import explain_validity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Anchor to the module file so the store resolves the same way regardless
# of CWD, both locally and inside the Docker container. Mirrors the same
# pattern as cache_manager.DEFAULT_CACHE_DIR.
DEFAULT_CONTRIBUTIONS_DIR = Path(
    os.environ.get(
        "BUILDING_CONTRIBUTIONS_DIR",
        str(Path(__file__).resolve().parent / "contributions"),
    )
)

MIN_AREA_SQM = 1.0
MAX_AREA_SQM = 50_000.0
MAX_VERTICES = 4096

ALLOWED_SOURCES = frozenset({"overture", "manual"})
# When deciding if an Overture polygon is "already in OSM", how much of its
# area must overlap any existing OSM building before we drop it as a
# candidate. 0.5 = half or more already mapped → skip.
OSM_OVERLAP_DROP_RATIO = 0.5


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ContributionError(ValueError):
    """Raised when a submitted polygon fails validation."""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class ContributionStore:
    """Folder-of-files persistence for approved building polygons."""

    def __init__(self, root: Path = DEFAULT_CONTRIBUTIONS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- read ------------------------------------------------------------
    def list(self) -> List[dict]:
        """Return all contributions, newest first."""
        with self._lock:
            paths = sorted(
                self.root.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return [self._load_file(p) for p in paths if not p.name.endswith(".tmp")]

    def get(self, contribution_id: str) -> Optional[dict]:
        if not _safe_id(contribution_id):
            return None
        p = self.root / f"{contribution_id}.json"
        if not p.exists():
            return None
        return self._load_file(p)

    def feature_collection(self) -> dict:
        return {"type": "FeatureCollection", "features": self.list()}

    def stats(self) -> dict:
        items = self.list()
        return {
            "count": len(items),
            "total_area_sqm": round(
                sum(
                    (it.get("properties") or {}).get("area_sqm", 0.0)
                    for it in items
                ),
                2,
            ),
        }

    # -- write -----------------------------------------------------------
    def add(
        self,
        feature: dict,
        *,
        notes: Optional[str] = None,
        author: Optional[str] = None,
        source: str = "manual",
        original_id: Optional[str] = None,
        edited: bool = False,
    ) -> dict:
        if source not in ALLOWED_SOURCES:
            raise ContributionError(
                f"source must be one of {sorted(ALLOWED_SOURCES)}"
            )
        polygon = _validate_polygon(feature)
        polygon = _shapely_orient(polygon, sign=1.0)  # RFC 7946: exterior CCW
        area_sqm = round(_area_sqm(polygon), 2)

        contribution_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "type": "Feature",
            "id": contribution_id,
            "geometry": mapping(polygon),
            "properties": {
                "id": contribution_id,
                "source": source,
                "original_id": original_id,
                "edited": bool(edited),
                "notes": notes,
                "author": author,
                "created_at": now,
                "area_sqm": area_sqm,
                "vertex_count": len(polygon.exterior.coords) - 1,
            },
        }
        self._write_atomic(contribution_id, record)
        return record

    def delete(self, contribution_id: str) -> bool:
        if not _safe_id(contribution_id):
            return False
        p = self.root / f"{contribution_id}.json"
        if p.exists():
            p.unlink()
            return True
        return False

    # -- internal --------------------------------------------------------
    def _load_file(self, p: Path) -> dict:
        try:
            with p.open() as f:
                return json.load(f)
        except Exception as e:
            return {"type": "Feature", "id": p.stem, "error": f"corrupt: {e}"}

    def _write_atomic(self, contribution_id: str, record: dict) -> None:
        p = self.root / f"{contribution_id}.json"
        tmp = p.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(record, f, indent=2)
        tmp.replace(p)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_polygon(feature: dict) -> Polygon:
    if not isinstance(feature, dict):
        raise ContributionError("feature must be an object")
    if feature.get("type") != "Feature":
        raise ContributionError("feature.type must be 'Feature'")

    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Polygon":
        raise ContributionError("geometry.type must be 'Polygon'")

    coords = geometry.get("coordinates") or []
    if not coords or not isinstance(coords[0], list):
        raise ContributionError("polygon needs a single exterior ring")
    # A closed ring needs 4+ points (first == last); 3 unique vertices min.
    if len(coords[0]) < 4:
        raise ContributionError("polygon needs at least 3 unique vertices")
    if len(coords[0]) > MAX_VERTICES:
        raise ContributionError(
            f"polygon has {len(coords[0])} vertices (> {MAX_VERTICES})"
        )

    try:
        poly = shape(geometry)
    except Exception as e:
        raise ContributionError(f"invalid geometry: {e}") from e

    if poly.geom_type != "Polygon":
        raise ContributionError(
            f"only Polygon geometries are accepted (got {poly.geom_type})"
        )
    if not poly.is_valid:
        fixed = poly.buffer(0)
        if (
            not fixed.is_valid
            or fixed.is_empty
            or fixed.geom_type != "Polygon"
        ):
            raise ContributionError(
                f"invalid polygon: {explain_validity(poly)}"
            )
        poly = fixed

    area_sqm = _area_sqm(poly)
    if area_sqm < MIN_AREA_SQM:
        raise ContributionError(
            f"polygon area {area_sqm:.2f} m² < min {MIN_AREA_SQM} m²"
        )
    if area_sqm > MAX_AREA_SQM:
        raise ContributionError(
            f"polygon area {area_sqm:.2f} m² > max {MAX_AREA_SQM} m²"
        )
    return poly


def _area_sqm(poly: Polygon) -> float:
    """Polygon area in square metres via the local UTM zone."""
    rep = poly.representative_point()
    lon, lat = rep.x, rep.y
    utm_zone = int((lon + 180) / 6) + 1
    epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
    to_utm = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    ).transform
    return float(transform(to_utm, poly).area)


def _safe_id(contribution_id: str) -> bool:
    """Reject IDs that could escape the contributions directory."""
    return (
        isinstance(contribution_id, str)
        and len(contribution_id) <= 64
        and all(c.isalnum() or c in "-_" for c in contribution_id)
    )


# ---------------------------------------------------------------------------
# Detection: Overture polygons not already in OSM
# ---------------------------------------------------------------------------
def overture_only_candidates(
    lat: float,
    lon: float,
    radius_meters: float,
    *,
    osm_all_buildings: bool = True,
) -> List[dict]:
    """Run Overture detection and return polygons not already in OSM.

    Reuses ``_fetch_and_filter_buildings`` (Overture + disk cache),
    ``get_osm_building_polygons(all_buildings=True)`` (OSM Overpass),
    and ``_building_union`` from ``visualization.py``. An Overture
    polygon is dropped if more than ``OSM_OVERLAP_DROP_RATIO`` of its
    area is already covered by an OSM building — that's "essentially
    already mapped".

    Each returned dict:
      {
        "id":            <overture id, str>,
        "geometry":      <GeoJSON Polygon (lon, lat)>,
        "area_sqm":      <float>,
        "vertex_count":  <int>,
      }
    """
    from ms_buildings import _fetch_and_filter_buildings
    from osm_query import get_osm_building_polygons
    from visualization import _building_union

    gdf = _fetch_and_filter_buildings(lat, lon, radius_meters)
    if gdf is None or len(gdf) == 0:
        return []

    osm_polys = get_osm_building_polygons(
        lat, lon, radius_meters, all_buildings=osm_all_buildings
    )
    osm_union = _building_union(osm_polys)

    candidates: List[dict] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            # Pick the largest sub-polygon so the user has a single editable shape.
            geom = max(geom.geoms, key=lambda g: g.area)
        if geom.geom_type != "Polygon":
            continue

        if osm_union is not None:
            try:
                overlap_area = geom.intersection(osm_union).area
            except Exception:
                overlap_area = 0.0
            if geom.area > 0 and (overlap_area / geom.area) > OSM_OVERLAP_DROP_RATIO:
                continue

        try:
            area = _area_sqm(geom)
        except Exception:
            continue
        if area < MIN_AREA_SQM or area > MAX_AREA_SQM:
            continue

        # Try to preserve the upstream Overture id so the user can trace
        # candidates back to the source row if they need to.
        overture_id = row.get("id") if hasattr(row, "get") else None
        candidate_id = str(overture_id) if overture_id else uuid.uuid4().hex[:12]

        candidates.append(
            {
                "id": candidate_id,
                "geometry": mapping(_shapely_orient(geom, sign=1.0)),
                "area_sqm": round(area, 2),
                "vertex_count": len(geom.exterior.coords) - 1,
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_default_store: Optional[ContributionStore] = None


def get_contribution_store() -> ContributionStore:
    """Return the process-wide :class:`ContributionStore` singleton."""
    global _default_store
    if _default_store is None:
        _default_store = ContributionStore()
    return _default_store
