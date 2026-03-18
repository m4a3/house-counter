FROM python:3.12-slim

# Install system dependencies for geopandas, shapely, pyproj, Pillow, and DuckDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libjpeg62-turbo-dev \
    libpng-dev \
    libfreetype6-dev \
    fonts-liberation \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY ms_buildings.py .
COPY osm_query.py .
COPY visualization.py .
COPY cache_manager.py .
COPY static/ static/

# Create building cache directory
RUN mkdir -p building_cache

EXPOSE 8008

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
