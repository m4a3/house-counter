# Front/Back Garden Classifier

Automatically distinguishes **front gardens** (facing the road) from **back gardens** (away from the road) using aerial imagery and OpenStreetMap data. Includes a **FastAPI server** for real-time lookups and a **batch precomputation** pipeline for processing entire suburbs.

![Example Output](output/garden_classification.png)

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Aerial Imagery (Google Tiles API)                          │
│     └─> RGB satellite/aerial photo stitched from map tiles     │
│     └─> Auto-zoom selection to stay within memory limits       │
├─────────────────────────────────────────────────────────────────┤
│  2. OpenStreetMap Data (FREE via Overpass API)                  │
│     └─> Building footprints, roads, driveways, address polygons│
│     └─> Property boundaries (fences, walls, hedges)            │
├─────────────────────────────────────────────────────────────────┤
│  3. Vegetation Detection (OpenCV + Excess Green Index)         │
│     └─> HSV colour segmentation + ExG confirmation             │
│     └─> Texture-based grass/tree canopy separation             │
│     └─> Image-based shade recovery (greenish hue + weak ExG)  │
├─────────────────────────────────────────────────────────────────┤
│  4. Geometric Classification                                    │
│     └─> STRtree spatial indexes for O(log n) road/driveway     │
│         nearest-neighbour lookups                              │
│     └─> Address-matching, driveway-matching, road-fallback     │
│     └─> Consensus correction + outlier detection               │
│     └─> Driveway override (near driveway = always front)       │
├─────────────────────────────────────────────────────────────────┤
│  5. Delivery Pin Placement                                      │
│     └─> One front + one back pin per building                  │
│     └─> 4-attempt fallback: classified → directional →         │
│         paved/driveway → relaxed ownership                     │
│     └─> Property border enforcement (configurable width)       │
│     └─> Lateral corridor constraint (building-width aligned)   │
│     └─> Tree canopy penalty, boundary penalty,                 │
│         claimed-area deduplication, shade recovery             │
│     └─> Post-processing: same-side correction,                 │
│         address-road distance fix, neighbor consistency        │
├─────────────────────────────────────────────────────────────────┤
│  6. Output                                                      │
│     └─> Segmentation mask, delivery pins, map visualisation    │
│     └─> REST API responses or cached precomputed results       │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add Google Tiles API Key

Create a `config_local.py` (git-ignored) with your API key:

```python
GOOGLE_TILES_API_KEY = "your-api-key-here"
```

> **Note:** You can run in demo mode without an API key using `--demo`

### 3. Run the API Server

```bash
python api.py --host 0.0.0.0 --port 8000
```

The interactive docs are available at `http://localhost:8000/docs`.

### 4. Run CLI Classification

```bash
# Default location, 500m radius
python main.py

# Custom location
python main.py --lat 53.3917 --lon -6.3878 --radius 300

# Demo mode (no API key required)
python main.py --demo
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/garden-pins` | Get front/back pins for a single address (lat/lon, coords string, or eircode) |
| `POST` | `/api/garden-pins/batch` | Get all delivery pins within a radius |
| `POST` | `/api/classify` | Classify a GPS coordinate as front/back garden |
| `POST` | `/api/precompute` | Precompute all pins for a large area |
| `GET` | `/api/cache/stats` | Get cache statistics |
| `DELETE` | `/api/cache` | Clear all caches |
| `POST` | `/api/shutdown` | Gracefully stop the server |

### Location input

Both `/api/garden-pins` and `/api/garden-pins/batch` accept coordinates in two forms:

```jsonc
// Separate fields
{ "lat": 53.3498, "lon": -6.2603 }

// Single string
{ "coords": "53.3498, -6.2603" }
```

The single-pin endpoint also accepts `"eircode": "D15 YXN8"` (geocoded via a 5-strategy cascade: Google Geocoding, Google Places, Nominatim postalcode, Nominatim free-text, routing key approximation).

### Map generation

Both endpoints support `"generate_map": true` to produce a PNG visualization of the pins. The map URL is returned in the response as `map_url` and served from `/static/`.

### Example requests

```bash
# Single house
curl -s -X POST "http://localhost:8000/api/garden-pins" \
  -H "Content-Type: application/json" \
  -d '{"coords": "53.380365, -6.386601", "generate_map": true}'

# All pins in a 500m radius
curl -s -X POST "http://localhost:8000/api/garden-pins/batch" \
  -H "Content-Type: application/json" \
  -d '{"coords": "53.3498, -6.2603", "radius_m": 500, "generate_map": true}'
```

## Batch Precomputation

For processing entire suburbs (hundreds of buildings), use the precompute endpoint or CLI tool:

```bash
python batch_precompute.py --lat 53.3917 --lon -6.3878 --radius 500
```

This fetches imagery once, classifies the full area, and stores pins for every building. Single-point lookups automatically fall back to the precompute cache when available, so subsequent API calls for any house within a precomputed area are near-instant.

## Output

Results are saved to the `output/` directory:

| File | Description |
|------|-------------|
| `garden_classification.png` | 4-panel comparison figure |
| `overlay.png` | Simple overlay on aerial image |
| `original.png` | Original aerial imagery |
| `maps/pins_map_*.png` | Generated delivery pin maps |

### Classification Colours

- **Green** = Front garden (facing road)
- **Blue** = Back garden (away from road)
- **Yellow outline** = Buildings
- **Red lines** = Roads

## Configuration

Edit `config.py` to customise:

```python
TARGET_LAT = 53.3917
TARGET_LON = -6.3878
ANALYSIS_RADIUS_M = 500

ZOOM_LEVEL = 19  # 18-20 recommended (auto-reduced for large radii)

# Green detection thresholds (HSV)
GREEN_HSV_LOWER = (35, 25, 25)
GREEN_HSV_UPPER = (85, 255, 255)

# Property border width (meters) — pins stay outside this zone
PROPERTY_BORDER_WIDTH_M = 3.0
```

## Project Structure

```
front-back-garden/
├── api.py                 # FastAPI server with REST endpoints
├── main.py                # CLI entry point
├── batch_precompute.py    # CLI batch precomputation tool
├── config.py              # Configuration and defaults
├── config_local.py        # Local overrides / API keys (git-ignored)
├── requirements.txt       # Python dependencies
├── src/
│   ├── tiles.py           # Google Tiles API fetcher + auto-zoom
│   ├── osm.py             # OpenStreetMap data fetcher (Overpass API)
│   ├── garden_detector.py # Vegetation detection + texture splitting
│   ├── classifier.py      # Front/back classification (STRtree indexes)
│   ├── fast_classifier.py # Single-point fast classification
│   ├── delivery_pins.py   # Delivery pin placement per building
│   ├── precompute.py      # Batch area precomputation manager
│   └── visualizer.py      # Output rendering
└── output/                # Generated results (git-ignored)
```

## Requirements

- Python 3.10+
- macOS / Linux
- ~2 GB disk space for dependencies
- Google Maps Platform API key (for real imagery)

## Accuracy Notes

The heuristic approach works well for:
- Typical suburban housing (semi-detached, terraced, detached)
- Clear road layouts with named streets
- Well-maintained gardens with grass
- Properties with driveways (strong front-garden signal)

May struggle with:
- Corner properties (multiple road frontages)
- Apartment complexes
- Properties with unusual layouts
- Very dense tree canopy obscuring ground
- Fully shaded gardens with no greenish hue remaining

## License

MIT
