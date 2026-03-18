"""
Visualization module

Creates output images showing the garden classification results,
including overlays, legends, and statistics.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Tuple, List, Optional, Dict
import geopandas as gpd

import config
from src.osm import geometry_to_pixel_coords
from src.classifier import GardenClassifier


def draw_buildings_on_image(
    image: np.ndarray,
    buildings: gpd.GeoDataFrame,
    geo_bounds: dict,
    color: Tuple[int, int, int] = (128, 128, 128),
    thickness: int = 2,
    fill: bool = False
) -> np.ndarray:
    """
    Draw building outlines on an image.
    
    Args:
        image: RGB image
        buildings: GeoDataFrame with building polygons
        geo_bounds: Geographic bounds dict
        color: RGB color for buildings
        thickness: Line thickness
        fill: Whether to fill buildings
        
    Returns:
        Image with buildings drawn
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    for _, building in buildings.iterrows():
        coords = geometry_to_pixel_coords(
            building.geometry,
            geo_bounds,
            (width, height)
        )
        
        if len(coords) >= 3:
            pts = np.array(coords, dtype=np.int32)
            if fill:
                cv2.fillPoly(result, [pts], color)
            else:
                cv2.polylines(result, [pts], True, color, thickness)
    
    return result


def draw_roads_on_image(
    image: np.ndarray,
    roads: gpd.GeoDataFrame,
    geo_bounds: dict,
    color: Tuple[int, int, int] = (200, 200, 200),
    thickness: int = 3
) -> np.ndarray:
    """
    Draw roads on an image.
    
    Args:
        image: RGB image
        roads: GeoDataFrame with road geometries
        geo_bounds: Geographic bounds dict
        color: RGB color for roads
        thickness: Line thickness
        
    Returns:
        Image with roads drawn
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    for _, road in roads.iterrows():
        coords = geometry_to_pixel_coords(
            road.geometry,
            geo_bounds,
            (width, height)
        )
        
        if len(coords) >= 2:
            pts = np.array(coords, dtype=np.int32)
            cv2.polylines(result, [pts], False, color, thickness)
    
    return result


def create_segmentation_visualization(
    image: np.ndarray,
    classification_mask: np.ndarray,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    geo_bounds: dict,
    stats: dict
) -> np.ndarray:
    """
    Create a comprehensive visualization of the garden classification.
    
    Args:
        image: Original aerial image
        classification_mask: Classification result
        buildings: GeoDataFrame with buildings
        roads: GeoDataFrame with roads
        geo_bounds: Geographic bounds
        stats: Classification statistics
        
    Returns:
        Visualization image with legend
    """
    height, width = image.shape[:2]
    
    # Create classification overlay
    overlay = image.copy()
    
    # Color front gardens (green)
    front_mask = classification_mask == GardenClassifier.FRONT_GARDEN
    overlay[front_mask] = (
        0.5 * overlay[front_mask] + 0.5 * np.array([100, 255, 100])
    ).astype(np.uint8)
    
    # Color back gardens (blue)
    back_mask = classification_mask == GardenClassifier.BACK_GARDEN
    overlay[back_mask] = (
        0.5 * overlay[back_mask] + 0.5 * np.array([100, 150, 255])
    ).astype(np.uint8)
    
    # Draw building outlines
    overlay = draw_buildings_on_image(
        overlay, buildings, geo_bounds,
        color=(255, 255, 0), thickness=2
    )
    
    # Draw roads
    overlay = draw_roads_on_image(
        overlay, roads, geo_bounds,
        color=(255, 100, 100), thickness=2
    )
    
    return overlay


def create_comparison_figure(
    original: np.ndarray,
    vegetation_mask: np.ndarray,
    classification_mask: np.ndarray,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    geo_bounds: dict,
    stats: dict,
    metadata: dict,
    output_path: str
):
    """
    Create a multi-panel comparison figure and save to file.
    
    Panels:
    1. Original aerial image
    2. Detected vegetation mask
    3. Front/back classification
    4. Final overlay with legend
    
    Args:
        original: Original aerial image
        vegetation_mask: Binary vegetation mask
        classification_mask: Classification result
        buildings: GeoDataFrame with buildings
        roads: GeoDataFrame with roads
        geo_bounds: Geographic bounds
        stats: Classification statistics
        metadata: Image metadata
        output_path: Path to save the figure
    """
    # Calculate figure size based on image size for better quality
    # Use larger figure for high-res images
    img_width, img_height = metadata["image_size"]
    # Scale figure size proportionally, max 32 inches per side
    scale = min(32 / (img_width / 100), 32 / (img_height / 100), 1.0)
    fig_size = max(16, min(32, img_width / 100 * scale * 2))
    
    fig, axes = plt.subplots(2, 2, figsize=(fig_size, fig_size))
    fig.suptitle(
        f"Front/Back Garden Classification\n"
        f"Center: ({metadata['center_lat']:.4f}, {metadata['center_lon']:.4f}) | "
        f"Radius: {metadata['radius_m']}m | Zoom: {metadata['zoom']}",
        fontsize=14
    )
    
    # Panel 1: Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Aerial Imagery")
    axes[0, 0].axis("off")
    
    # Panel 2: Vegetation mask
    axes[0, 1].imshow(vegetation_mask, cmap="Greens")
    axes[0, 1].set_title(f"Detected Vegetation ({stats['total_garden_pixels']:,} pixels)")
    axes[0, 1].axis("off")
    
    # Panel 3: Classification mask
    class_display = np.zeros((*classification_mask.shape, 3), dtype=np.uint8)
    class_display[classification_mask == GardenClassifier.FRONT_GARDEN] = [100, 255, 100]
    class_display[classification_mask == GardenClassifier.BACK_GARDEN] = [100, 150, 255]
    axes[1, 0].imshow(class_display)
    axes[1, 0].set_title(
        f"Classification: Front ({stats['front_percentage']:.1f}%) / "
        f"Back ({stats['back_percentage']:.1f}%)"
    )
    axes[1, 0].axis("off")
    
    # Panel 4: Final overlay
    overlay = create_segmentation_visualization(
        original, classification_mask,
        buildings, roads, geo_bounds, stats
    )
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Final Result with Buildings & Roads")
    axes[1, 1].axis("off")
    
    # Add legend
    legend_elements = [
        Patch(facecolor=(0.4, 1.0, 0.4), label="Front Garden"),
        Patch(facecolor=(0.4, 0.6, 1.0), label="Back Garden"),
        Patch(facecolor=(1.0, 1.0, 0.0), edgecolor="black", label="Buildings"),
        Patch(facecolor=(1.0, 0.4, 0.4), label="Roads"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=12,
        framealpha=0.9
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figure at high DPI for better quality
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def create_simple_overlay(
    image: np.ndarray,
    classification_mask: np.ndarray,
    output_path: str,
    alpha: float = 0.5
):
    """
    Create a simple overlay image without matplotlib.
    
    This saves the FULL RESOLUTION overlay - use this for detailed inspection.
    
    Colors:
    - Green = Front garden
    - Blue = Back garden  
    - Red = Unknown/ambiguous
    
    Args:
        image: Original RGB image
        classification_mask: Classification result
        output_path: Path to save the image
        alpha: Overlay transparency
    """
    overlay = image.copy()
    
    # Front gardens - green
    front_mask = classification_mask == GardenClassifier.FRONT_GARDEN
    overlay[front_mask] = (
        (1 - alpha) * overlay[front_mask] + alpha * np.array([100, 255, 100])
    ).astype(np.uint8)
    
    # Back gardens - blue
    back_mask = classification_mask == GardenClassifier.BACK_GARDEN
    overlay[back_mask] = (
        (1 - alpha) * overlay[back_mask] + alpha * np.array([100, 150, 255])
    ).astype(np.uint8)
    
    # Unknown/ambiguous - red
    unknown_mask = classification_mask == GardenClassifier.UNKNOWN
    overlay[unknown_mask] = (
        (1 - alpha) * overlay[unknown_mask] + alpha * np.array([255, 100, 100])
    ).astype(np.uint8)
    
    # Save using PIL at maximum quality
    from PIL import Image
    img = Image.fromarray(overlay)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    img.save(output_path, quality=95)


def create_fullres_classification(
    image: np.ndarray,
    classification_mask: np.ndarray,
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    geo_bounds: dict,
    output_path: str,
    alpha: float = 0.4
):
    """
    Create a full-resolution classification overlay with buildings and roads.
    
    This is the highest quality output - no downsampling.
    
    Colors:
    - Green = Front garden
    - Blue = Back garden
    - Red = Unknown/ambiguous
    
    Args:
        image: Original RGB image (full resolution)
        classification_mask: Classification result
        buildings: GeoDataFrame with buildings
        roads: GeoDataFrame with roads
        geo_bounds: Geographic bounds
        output_path: Path to save
        alpha: Overlay transparency
    """
    from PIL import Image
    
    height, width = image.shape[:2]
    overlay = image.copy()
    
    # Apply classification colors
    front_mask = classification_mask == GardenClassifier.FRONT_GARDEN
    back_mask = classification_mask == GardenClassifier.BACK_GARDEN
    unknown_mask = classification_mask == GardenClassifier.UNKNOWN
    
    overlay[front_mask] = (
        (1 - alpha) * overlay[front_mask] + alpha * np.array([100, 255, 100])
    ).astype(np.uint8)
    
    overlay[back_mask] = (
        (1 - alpha) * overlay[back_mask] + alpha * np.array([100, 150, 255])
    ).astype(np.uint8)
    
    overlay[unknown_mask] = (
        (1 - alpha) * overlay[unknown_mask] + alpha * np.array([255, 100, 100])
    ).astype(np.uint8)
    
    # Draw building outlines
    overlay = draw_buildings_on_image(
        overlay, buildings, geo_bounds,
        color=(255, 255, 0), thickness=2
    )
    
    # Draw roads
    overlay = draw_roads_on_image(
        overlay, roads, geo_bounds,
        color=(255, 100, 100), thickness=2
    )
    
    # Save at full resolution
    img = Image.fromarray(overlay)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    img.save(output_path, quality=95)


def print_statistics(stats: dict, metadata: dict):
    """
    Print classification statistics to console.
    """
    print("\n" + "=" * 60)
    print("GARDEN CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Location: ({metadata['center_lat']:.4f}, {metadata['center_lon']:.4f})")
    print(f"Radius: {metadata['radius_m']}m")
    print(f"Image size: {metadata['image_size'][0]} x {metadata['image_size'][1]} pixels")
    print("-" * 60)
    print(f"Total garden pixels: {stats['total_garden_pixels']:,}")
    print(f"Front garden pixels: {stats['front_garden_pixels']:,} ({stats['front_percentage']:.1f}%)")
    print(f"Back garden pixels:  {stats['back_garden_pixels']:,} ({stats['back_percentage']:.1f}%)")
    print("=" * 60 + "\n")
