# House Counter API

Count buildings within a given radius using Microsoft Building Footprints (ML-derived from satellite imagery), with optional map visualization using Google satellite tiles. Also includes an OSM contributor UI for surfacing and filing OSM blind spots.

## Features

- **Accurate Building Detection**: Uses Microsoft's ML-derived building footprints via Overture Maps
- **No API Key Required**: All data sources are publicly accessible
- **Visual Output**: Generate map images with Google satellite tiles as background
- **Configurable Zoom**: Choose detail level for map output (trades speed for resolution)
- **OSM Contributor UI**: Browser-based tool at `/contribute` for editing and saving OSM-missing buildings as GeoJSON

## Installation

### Docker (recommended)

```bash
docker compose up --build
```

The server runs on `http://localhost:8008`. The Overture parquet cache and approved OSM contributions are persisted in named Docker volumes (`building_cache` and `contributions`) so both survive container rebuilds.

To pass environment variables (e.g. `GOOGLE_TILES_API_KEY`), create a `.env` file in the project root before starting the container.

#### Running from ECR (CI-built image)

To use the image built and pushed by the GitHub Actions pipeline (e.g. on EC2):

```bash
export ECR_REGISTRY=123456789012.dkr.ecr.eu-west-1.amazonaws.com   # your account + region
export IMAGE_TAG=main   # or a semver tag e.g. v1.0.0, or commit SHA
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $ECR_REGISTRY
docker compose -f docker-compose.ecr.yml up -d
```

### Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The server runs on `http://localhost:8008`

## API Endpoints

### `GET /count`

Count buildings within a radius. **Optimized for speed** - no map generation.

```bash
curl "http://localhost:8008/count?lat=36.060345&lon=-95.816314&radius_km=3"
```

**Response:**
```json
{
  "latitude": 36.060345,
  "longitude": -95.816314,
  "radius_km": 3.0,
  "building_count": 11308,
  "total_area_sqm": 3921956.23,
  "avg_building_area_sqm": 346.83,
  "message": "Found 11308 buildings within 3.0km"
}
```

### `GET /map`

Get a PNG map image showing buildings in the area.

```bash
curl "http://localhost:8008/map?lat=36.060345&lon=-95.816314&radius_km=3" -o map.png

# High resolution (zoom 17, ~5 min)
curl "http://localhost:8008/map?lat=36.060345&lon=-95.816314&radius_km=3&zoom=17" -o map_hires.png
```

### `GET /count-with-map`

Count buildings and save map image to disk.

```bash
curl "http://localhost:8008/count-with-map?lat=36.060345&lon=-95.816314&radius_km=3"
```

### `GET /zoom-info`

Get zoom level options and timing estimates.

```bash
curl "http://localhost:8008/zoom-info?radius_km=3"
```

### `GET /compare`

Compare building counts from Microsoft vs OpenStreetMap for validation testing.

```bash
curl "http://localhost:8008/compare?lat=36.060345&lon=-95.816314&radius_km=3"
```

**Response:**
```json
{
  "latitude": 36.060345,
  "longitude": -95.816314,
  "radius_km": 3.0,
  "microsoft": {
    "building_count": 11308,
    "total_area_sqm": 3921956.23,
    "source": "Microsoft Building Footprints (ML from satellite)"
  },
  "osm": {
    "building_count": 352,
    "source": "OpenStreetMap (crowdsourced)"
  },
  "comparison": "Microsoft found +10956 more buildings (+3112.5%)"
}
```

### `GET /compare-with-map`

Compare counts and generate a map image with a polygon set-difference overlay so agreement vs blind spots is unambiguous:

- **green** — MS ∩ OSM (both sources agree)
- **red** — MS − OSM (OSM blind spot — Microsoft has the building but OSM doesn't)
- **blue** — OSM − MS (MS missed it — OSM has the building but Microsoft doesn't)

```bash
curl "http://localhost:8008/compare-with-map?lat=36.060345&lon=-95.816314&radius_km=3"
```

## OSM Contributor UI

A browser-based tool for fixing OSM blind spots without leaving the service. Browse to:

```
http://localhost:8008/contribute
```

Workflow: pan to an area, click **Detect** to load existing OSM buildings (blue) and Overture polygons that are not yet in OSM (orange candidates). Click a candidate to drag its vertices for a closer fit, or use **Approve all** to bulk-promote them. Approved polygons turn green and are persisted as one GeoJSON file per polygon under `contributions/`. **Export GeoJSON** downloads the full set as a single `FeatureCollection` ready to be reviewed and imported into OSM.

Endpoints:

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/contribute` | Serves the contributor UI |
| `GET` | `/contribute/osm-buildings?lat&lon&radius_km` | Existing OSM buildings as a GeoJSON `FeatureCollection` |
| `GET` | `/contribute/overture-candidates?lat&lon&radius_km&only_missing=true` | Overture polygons whose area is < 50 % already in OSM. `only_missing=false` returns every Overture polygon in the radius. |
| `GET` | `/contribute/contributions` | List saved contributions newest-first + stats |
| `POST` | `/contribute/contributions` | Validate + persist a single polygon |
| `POST` | `/contribute/contributions/bulk` | Validate + persist many polygons in one round-trip |
| `POST` | `/contribute/contributions/bulk-delete` | Remove many contributions in one call |
| `DELETE` | `/contribute/contributions/{id}` | Remove one |
| `DELETE` | `/contribute/contributions?confirm=true` | Remove every contribution |
| `GET` | `/contribute/contributions/export.geojson` | Single merged `FeatureCollection`, `Content-Disposition: attachment` |

Radius is capped at 3 km per call so a single Overpass / Overture round-trip stays under ~30 s on a typical link. The OSM lookup is cached in-memory for 5 minutes per `(lat, lon, radius, all_buildings)` tuple so repeated detection in the same area stays fast and survives short Overpass outages. When Overpass is unavailable the candidate endpoint still returns every Overture polygon in the radius with `meta.osm_available: false` so the UI can warn rather than 500.

## Zoom Levels

For a 3km radius query:

| Zoom | Tiles | Image Size | Time |
|------|-------|------------|------|
| 14 (default) | 25 | 1,280 px | ~5 sec |
| 15 | 100 | 2,560 px | ~15 sec |
| 16 | 400 | 5,120 px | ~1 min |
| 17 | 1,600 | 10,240 px | ~5 min |
| 18 | 6,400 | 20,480 px | ~20 min |

## Data Sources

- **Building Data**: [Overture Maps](https://overturemaps.org/) (includes Microsoft Building Footprints)
- **Map Tiles**: Google Maps satellite imagery

## How It Works

1. **Building Count**: Queries pre-computed building polygons from Overture Maps (Microsoft's ML-derived footprints stored on AWS S3)
2. **Map Generation**: Downloads Google satellite tiles, overlays building polygons in red, adds search radius circle

The building detection was done offline by Microsoft using computer vision on satellite imagery. This API simply queries that pre-computed dataset.

## License

MIT
