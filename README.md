# Property APIs — Monorepo

Two Python/FastAPI services deployed together behind an nginx reverse proxy.

## Services

| Service | Path Prefix | Port | Description |
|---|---|---|---|
| [house-counter](house-counter/README.md) | `/buildings/` | 8008 | Count buildings within a radius using Microsoft Building Footprints, plus an OSM contributor UI at `/buildings/contribute` |
| [front-back-garden](front-back-garden/README.md) | `/garden/` | 8000 | Classify front/back gardens and place delivery pins using aerial imagery |

## Quick Start

```bash
# Copy and fill in your API keys
cp .env.example .env

# Build and run all services locally
docker compose up --build
```

- House Counter API: `http://localhost/buildings/count?lat=53.38&lon=-6.38&radius_km=1`
- House Counter contributor UI: `http://localhost/buildings/contribute`
- Garden Classifier API: `http://localhost/garden/docs`

## Environment Variables

Create a `.env` file at the repo root:

```
GOOGLE_TILES_API_KEY=your-key-here
```

Both services read `GOOGLE_TILES_API_KEY` from this file.

## Running on EC2 (from ECR)

```bash
export ECR_REGISTRY=245475270127.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_TAG=main   # or a semver tag e.g. v1.0.0

aws ecr get-login-password --region eu-west-1 \
  | docker login --username AWS --password-stdin $ECR_REGISTRY

docker compose -f docker-compose.ecr.yml up -d
```

## Repository Structure

```
/
├── house-counter/          # Building count service
│   ├── main.py
│   ├── Dockerfile
│   └── ...
├── front-back-garden/      # Garden classifier service
│   ├── api.py
│   ├── Dockerfile
│   └── ...
├── nginx/
│   └── nginx.conf          # Reverse proxy — routes by path prefix
├── docker-compose.yml      # Local development (builds from source)
├── docker-compose.ecr.yml  # Production (pulls from ECR)
└── .github/workflows/
    └── ci_build.yml        # Builds and pushes both images to ECR
```

## Updating front-back-garden (git subtree)

The `front-back-garden/` directory is managed as a git subtree. To pull upstream changes:

```bash
git subtree pull --prefix front-back-garden https://github.com/tiernan-manna/front-back-garden.git main --squash
```
