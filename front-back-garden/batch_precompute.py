#!/usr/bin/env python3
"""
Batch Precomputation CLI

Command-line tool for precomputing garden classification data for large areas.
This is useful for:
- Seeding the cache before going live in an area
- Processing entire cities/regions in advance
- Optimizing tile usage by batching requests

Usage:
    python batch_precompute.py --lat 53.375 --lon -6.384 --radius 5000
    python batch_precompute.py --lat 53.375 --lon -6.384 --radius 5000 --tile-source manna
    python batch_precompute.py --clear-cache
    python batch_precompute.py --stats

Examples:
    # Precompute Blanchardstown area (5km radius)
    python batch_precompute.py --lat 53.375 --lon -6.384 --radius 5000

    # Precompute Dublin city center (10km radius) using Manna tiles
    python batch_precompute.py --lat 53.349 --lon -6.260 --radius 10000 --tile-source manna

    # Check cache statistics
    python batch_precompute.py --stats
"""

import argparse
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tiles import TileSource
from src.precompute import PrecomputeManager


def main():
    parser = argparse.ArgumentParser(
        description="Batch precompute garden classification data for large areas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Precompute 5km radius around Blanchardstown
    python batch_precompute.py --lat 53.375 --lon -6.384 --radius 5000

    # Use Manna tiles instead of Google
    python batch_precompute.py --lat 53.375 --lon -6.384 --radius 5000 --tile-source manna

    # Check cache statistics
    python batch_precompute.py --stats

    # Clear the cache
    python batch_precompute.py --clear-cache
        """
    )
    
    # Location arguments
    parser.add_argument(
        "--lat",
        type=float,
        help="Center latitude"
    )
    parser.add_argument(
        "--lon",
        type=float,
        help="Center longitude"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1000,
        help="Radius in meters (default: 1000)"
    )
    
    # Tile source
    parser.add_argument(
        "--tile-source",
        choices=["google", "manna", "auto"],
        default="auto",
        help="Tile source to use (default: auto)"
    )
    
    # Processing options
    parser.add_argument(
        "--zoom",
        type=int,
        default=19,
        help="Zoom level (default: 19)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process sequentially instead of in parallel"
    )
    
    # Cache management
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics and exit"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the precompute cache and exit"
    )
    
    args = parser.parse_args()
    
    # Map tile source string to enum
    source_map = {
        "google": TileSource.GOOGLE,
        "manna": TileSource.MANNA,
        "auto": TileSource.AUTO
    }
    tile_source = source_map[args.tile_source]
    
    # Create manager
    manager = PrecomputeManager(
        tile_source=tile_source,
        zoom=args.zoom,
        max_workers=args.workers
    )
    
    # Handle cache operations
    if args.stats:
        stats = manager.get_cache_stats()
        print("\n📊 Precompute Cache Statistics")
        print("=" * 40)
        print(f"Total chunks:  {stats['total_chunks']}")
        print(f"Total pins:    {stats['total_pins']}")
        print(f"Cache size:    {stats['cache_size_mb']:.2f} MB")
        print(f"Cache dir:     {stats['cache_dir']}")
        print("=" * 40 + "\n")
        return 0
    
    if args.clear_cache:
        manager.clear_cache()
        print("✅ Cache cleared")
        return 0
    
    # Validate location arguments
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon are required for precomputation")
    
    # Run precomputation
    print("\n" + "=" * 60)
    print("🌿 GARDEN CLASSIFICATION BATCH PRECOMPUTATION")
    print("=" * 60)
    print(f"Center:      ({args.lat:.6f}, {args.lon:.6f})")
    print(f"Radius:      {args.radius}m")
    print(f"Zoom:        {args.zoom}")
    print(f"Tile source: {args.tile_source}")
    print(f"Workers:     {args.workers}")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    try:
        summary = manager.precompute_area(
            center_lat=args.lat,
            center_lon=args.lon,
            radius_m=args.radius,
            parallel=not args.sequential,
            show_progress=True
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("✅ PRECOMPUTATION COMPLETE")
        print("=" * 60)
        print(f"Chunks processed: {summary['chunks_successful']}/{summary['chunks_total']}")
        print(f"Delivery pins:    {summary['total_pins']}")
        print(f"Buildings:        {summary['total_buildings']}")
        print(f"Total time:       {elapsed:.1f}s")
        print(f"Tile source:      {summary['tile_source']}")
        print("=" * 60 + "\n")
        
        if summary['chunks_failed'] > 0:
            print(f"⚠️  {summary['chunks_failed']} chunks failed to process")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
