"""
Configuration for Front/Back Garden Classifier

To use real imagery, create a config_local.py file with your API key:
    GOOGLE_TILES_API_KEY = "your-api-key-here"

Alternatively, set the GOOGLE_TILES_API_KEY environment variable (used in Docker).
"""

import os

# =============================================================================
# GOOGLE TILES API CONFIGURATION
# =============================================================================

# Try to load API key from config_local.py (gitignored), fall back to env var
try:
    from config_local import GOOGLE_TILES_API_KEY
except ImportError:
    GOOGLE_TILES_API_KEY = os.environ.get("GOOGLE_TILES_API_KEY", "")

# Google Tiles API endpoint (Map Tiles API - Satellite)
GOOGLE_TILES_URL = "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}"

# =============================================================================
# MANNA TILES API CONFIGURATION (Alternative tile source)
# =============================================================================

# Try to load Manna config from config_local.py (gitignored)
try:
    from config_local import MANNA_TILES_URL, MANNA_API_KEY
except ImportError:
    MANNA_TILES_URL = os.environ.get("MANNA_TILES_URL")
    MANNA_API_KEY = os.environ.get("MANNA_API_KEY")

# =============================================================================
# TARGET LOCATION - Blanchardstown, Dublin 15
# =============================================================================

# Approximate coordinates for Blanchardstown
# These will be refined by geocoding on first run
TARGET_ADDRESS = "Blanchardstown, Dublin 15, Ireland"
TARGET_LAT = 53.375142480929455  # Latitude
TARGET_LON = -6.383740530507169  # Longitude

# Analysis radius in meters
ANALYSIS_RADIUS_M = 500

# =============================================================================
# IMAGERY SETTINGS
# =============================================================================

# Zoom level for aerial imagery
# 18 = ~0.6m/pixel (good balance of detail and coverage)
# 19 = ~0.3m/pixel (more detail, more tiles needed)
# 20 = ~0.15m/pixel (highest detail, many tiles)
ZOOM_LEVEL = 19

# Tile size in pixels (Google standard)
TILE_SIZE = 256

# =============================================================================
# GARDEN DETECTION SETTINGS (HSV color thresholds for green detection)
# =============================================================================

# HSV thresholds for detecting green vegetation
# These work well for Irish/UK gardens with grass
GREEN_HSV_LOWER = (35, 25, 25)   # Hue, Saturation, Value
GREEN_HSV_UPPER = (85, 255, 255)

# Minimum area (in pixels) to consider as a garden
MIN_GARDEN_AREA_PX = 100

# Minimum border width between adjacent houses (meters).
# Pins cannot be placed within half this distance of a property boundary edge.
# This prevents pins from landing in the border zone between houses.
PROPERTY_BORDER_WIDTH_M = 3.0

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = "output"

# Colors for visualization (BGR format for OpenCV)
FRONT_GARDEN_COLOR = (0, 255, 0)    # Green
BACK_GARDEN_COLOR = (255, 0, 0)     # Blue
BUILDING_COLOR = (128, 128, 128)    # Gray
ROAD_COLOR = (200, 200, 200)        # Light gray
