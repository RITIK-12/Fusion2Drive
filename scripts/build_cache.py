#!/usr/bin/env python3
"""
Build cache for Waymo dataset.

This script preprocesses the Waymo dataset and builds a cache
for faster training iteration.

Usage:
    python scripts/build_cache.py --data_dir data/waymo --cache_dir data/cache
    python scripts/build_cache.py --data_dir data/waymo --cache_dir data/cache --split training
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wod_fusion.data import CacheBuilder
from wod_fusion.data.cache import CacheConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(description="Build Waymo dataset cache")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to Waymo dataset root (containing training/, validation/)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to output cache directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "training", "validation"],
        help="Split to build cache for",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 704],
        help="Target image size (H W)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=150000,
        help="Maximum LiDAR points per frame",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild existing cache",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Building Waymo dataset cache")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Cache directory: {args.cache_dir}")
    
    # Determine splits to process
    splits = ["training", "validation"] if args.split == "all" else [args.split]
    
    for split in splits:
        logger.info(f"\nProcessing {split} split...")
        
        input_dir = Path(args.data_dir) / split
        output_dir = Path(args.cache_dir) / split
        
        if not input_dir.exists():
            logger.warning(f"Split directory not found: {input_dir}")
            continue
        
        if output_dir.exists() and not args.force:
            logger.info(f"Cache already exists: {output_dir}")
            logger.info("Use --force to rebuild")
            continue
        
        # Create config and builder for this split
        config = CacheConfig(
            image_size=tuple(args.image_size),
            max_points=args.max_points,
            num_workers=args.num_workers,
        )
        
        builder = CacheBuilder(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            config=config,
        )
        
        builder.build()
        logger.info(f"Completed {split}")
    
    logger.info("\nCache build complete!")
    
    # Show cache statistics
    cache_dir = Path(args.cache_dir)
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        logger.info(f"Total cache size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
