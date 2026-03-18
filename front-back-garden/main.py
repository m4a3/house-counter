#!/usr/bin/env python3
"""
Front/Back Garden Classifier

Automatically distinguishes front and back gardens from aerial imagery
using geometric heuristics and OpenStreetMap data.

Usage:
    python main.py                    # Use default location (67 Clonsilla Rd)
    python main.py --lat 53.39 --lon -6.39 --radius 300
    python main.py --demo             # Run without Google API key (placeholder image)

Requirements:
    - Python 3.10+
    - Dependencies: pip install -r requirements.txt
    - Google Tiles API key (add to config.py) for real imagery
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.tiles import fetch_area_image, create_placeholder_image
from src.osm import fetch_all_osm_data, geometry_to_pixel_coords
from src.garden_detector import detect_green_areas, exclude_buildings_from_mask, exclude_roads_from_mask
from src.classifier import GardenClassifier
from src.visualizer import (
    create_comparison_figure,
    create_simple_overlay,
    create_fullres_classification,
    print_statistics,
)


def run_classification(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    zoom: int,
    demo_mode: bool = False,
    output_dir: str = None,
    use_cache: bool = True
) -> dict:
    """
    Run the full garden classification pipeline.
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_m: Analysis radius in meters
        zoom: Zoom level for imagery
        demo_mode: If True, use placeholder image (no API key needed)
        output_dir: Output directory for results
        use_cache: If True, use cached tiles if available
        
    Returns:
        Dict with classification results and statistics
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("FRONT/BACK GARDEN CLASSIFIER")
    print("=" * 60)
    print(f"Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"Radius: {radius_m}m | Zoom: {zoom}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    # =========================================================================
    # Step 1: Fetch aerial imagery
    # =========================================================================
    print("\n[Step 1/5] Fetching aerial imagery...")
    
    if demo_mode or not config.GOOGLE_TILES_API_KEY:
        if not demo_mode:
            print("⚠️  No Google API key found in config.py")
            print("    Running in demo mode with placeholder image...")
            print("    Add your API key to config.py for real imagery.\n")
        image, metadata = create_placeholder_image(center_lat, center_lon, radius_m, zoom)
    else:
        try:
            image, metadata = fetch_area_image(center_lat, center_lon, radius_m, zoom, use_cache=use_cache)
        except Exception as e:
            print(f"Error fetching imagery: {e}")
            print("Falling back to placeholder image...")
            image, metadata = create_placeholder_image(center_lat, center_lon, radius_m, zoom)
    
    # =========================================================================
    # Step 2: Fetch OSM data (buildings and roads)
    # =========================================================================
    print("\n[Step 2/5] Fetching OpenStreetMap data...")
    
    osm_data = fetch_all_osm_data(center_lat, center_lon, radius_m)
    buildings = osm_data["buildings"]
    roads = osm_data["roads"]
    driveways = osm_data.get("driveways", None)
    exclusion_zones = osm_data.get("exclusion_zones", None)
    property_boundaries = osm_data.get("property_boundaries", None)
    address_polygons = osm_data.get("address_polygons", None)
    
    if driveways is not None:
        print(f"    Driveways: {len(driveways)}")
    if exclusion_zones is not None:
        print(f"    Exclusion zones: {len(exclusion_zones)}")
    if property_boundaries is not None:
        print(f"    Property boundaries (walls/fences): {len(property_boundaries)}")
    if address_polygons is not None:
        print(f"    Address polygons: {len(address_polygons)}")
    
    # =========================================================================
    # Step 3: Detect vegetation (green areas)
    # =========================================================================
    print("\n[Step 3/5] Detecting vegetation...")
    
    vegetation_mask, contours = detect_green_areas(image)
    
    # Convert building and road geometries to pixel coordinates
    geo_bounds = metadata["geo_bounds"]
    image_size = metadata["image_size"]
    
    from tqdm import tqdm
    
    building_polys_px = []
    for _, building in tqdm(buildings.iterrows(), total=len(buildings), desc="Converting buildings"):
        coords = geometry_to_pixel_coords(building.geometry, geo_bounds, image_size)
        if coords:
            building_polys_px.append(coords)
    
    road_lines_px = []
    for _, road in tqdm(roads.iterrows(), total=len(roads), desc="Converting roads"):
        coords = geometry_to_pixel_coords(road.geometry, geo_bounds, image_size)
        if coords:
            road_lines_px.append(coords)
    
    # Exclude buildings and roads from vegetation mask
    print("Excluding buildings and roads from vegetation...")
    vegetation_mask = exclude_buildings_from_mask(vegetation_mask, building_polys_px)
    vegetation_mask = exclude_roads_from_mask(vegetation_mask, road_lines_px, road_width_px=8)
    
    veg_pixels = np.sum(vegetation_mask > 0)
    print(f"    Vegetation pixels: {veg_pixels:,}")
    
    # =========================================================================
    # Step 4: Classify gardens as front/back
    # =========================================================================
    print("\n[Step 4/5] Classifying front/back gardens...")
    
    classifier = GardenClassifier(
        buildings=buildings,
        roads=roads,
        geo_bounds=geo_bounds,
        image_size=image_size,
        center_lat=center_lat,
        center_lon=center_lon,
        driveways=driveways,
        exclusion_zones=exclusion_zones,
        property_boundaries=property_boundaries,
        address_polygons=address_polygons
    )
    
    # Classify with sampling for speed
    classification_mask = classifier.classify_mask(vegetation_mask, sample_step=4)
    
    # Get statistics
    stats = classifier.get_classification_stats(classification_mask)
    
    # =========================================================================
    # Step 5: Generate output visualizations
    # =========================================================================
    print("\n[Step 5/5] Generating visualizations...")
    
    viz_steps = tqdm(total=4, desc="Saving outputs")
    
    # Main comparison figure (for overview)
    comparison_path = os.path.join(output_dir, "garden_classification.png")
    create_comparison_figure(
        original=image,
        vegetation_mask=vegetation_mask,
        classification_mask=classification_mask,
        buildings=buildings,
        roads=roads,
        geo_bounds=geo_bounds,
        stats=stats,
        metadata=metadata,
        output_path=comparison_path
    )
    viz_steps.update(1)
    
    # Full-resolution detailed output (best for zooming in)
    fullres_path = os.path.join(output_dir, "classification_fullres.png")
    create_fullres_classification(
        image=image,
        classification_mask=classification_mask,
        buildings=buildings,
        roads=roads,
        geo_bounds=geo_bounds,
        output_path=fullres_path
    )
    viz_steps.update(1)
    
    # Simple overlay (full resolution without buildings/roads)
    overlay_path = os.path.join(output_dir, "overlay.png")
    create_simple_overlay(image, classification_mask, overlay_path)
    viz_steps.update(1)
    
    # Save the original image
    from PIL import Image as PILImage
    original_path = os.path.join(output_dir, "original.png")
    PILImage.fromarray(image).save(original_path)
    viz_steps.update(1)
    viz_steps.close()
    
    # Print statistics
    print_statistics(stats, metadata)
    
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.1f} seconds\n")
    
    return {
        "image": image,
        "vegetation_mask": vegetation_mask,
        "classification_mask": classification_mask,
        "buildings": buildings,
        "roads": roads,
        "metadata": metadata,
        "stats": stats,
    }


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Classify front and back gardens from aerial imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Use default location
    python main.py --demo                    # Run without API key
    python main.py --lat 53.39 --lon -6.39   # Custom coordinates
    python main.py --radius 300              # Smaller radius
        """
    )
    
    parser.add_argument(
        "--lat",
        type=float,
        default=config.TARGET_LAT,
        help=f"Center latitude (default: {config.TARGET_LAT})"
    )
    
    parser.add_argument(
        "--lon",
        type=float,
        default=config.TARGET_LON,
        help=f"Center longitude (default: {config.TARGET_LON})"
    )
    
    parser.add_argument(
        "--radius",
        type=float,
        default=config.ANALYSIS_RADIUS_M,
        help=f"Analysis radius in meters (default: {config.ANALYSIS_RADIUS_M})"
    )
    
    parser.add_argument(
        "--zoom",
        type=int,
        default=config.ZOOM_LEVEL,
        help=f"Zoom level for imagery (default: {config.ZOOM_LEVEL})"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with placeholder image (no API key needed)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-fetch tiles even if cached"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the tile cache and exit"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=config.OUTPUT_DIR,
        help=f"Output directory (default: {config.OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    # Handle cache clear
    if args.clear_cache:
        from src.tiles import clear_cache
        clear_cache()
        return 0
    
    # Run classification
    try:
        result = run_classification(
            center_lat=args.lat,
            center_lon=args.lon,
            radius_m=args.radius,
            zoom=args.zoom,
            demo_mode=args.demo,
            output_dir=args.output,
            use_cache=not args.no_cache
        )
        
        # Print summary
        stats = result["stats"]
        if stats["total_garden_pixels"] > 0:
            print("✅ Classification complete!")
            print(f"   Front gardens: {stats['front_percentage']:.1f}%")
            print(f"   Back gardens:  {stats['back_percentage']:.1f}%")
        else:
            print("⚠️  No gardens detected. This might be because:")
            print("   - Using placeholder image (add API key for real imagery)")
            print("   - Area has no vegetation")
            print("   - Zoom level too low")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
