#!/usr/bin/env python3
"""
Download Waymo Open Dataset with automatic subset selection.

This script downloads the Waymo Open Dataset from Google Cloud Storage,
pulling sequences from the official training/validation/testing splits.

Usage:
    # Download 500 sequences total, distributed across train/val/test (recommended for dev)
    python scripts/download_waymo.py --output-dir data/waymo --num-sequences 500
    
    # Download using presets
    python scripts/download_waymo.py --output-dir data/waymo --subset medium
    
    # Download full official splits
    python scripts/download_waymo.py --output-dir data/waymo --subset full
    
    # Download only training split (full)
    python scripts/download_waymo.py --output-dir data/waymo --subset full --splits training
"""

import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Preset configurations (training, validation, testing counts)
PRESETS = {
    "tiny": {"training": 30, "validation": 15, "testing": 5, "description": "Quick testing (~5GB)"},
    "small": {"training": 70, "validation": 20, "testing": 10, "description": "Light development (~10GB)"},
    "medium": {"training": 500, "validation": 150, "testing": 50, "description": "Full development (~70GB)"},
    "large": {"training": 700, "validation": 150, "testing": 50, "description": "Extended training (~100GB)"},
    "full": {"training": None, "validation": None, "testing": None, "description": "Complete dataset (~1TB)"},
}

# GCS bucket paths
WAYMO_V2_BUCKET = "waymo_open_dataset_v_2_0_1"
WAYMO_V1_BUCKET = "waymo_open_dataset_v_1_4_3"

# Official split names
OFFICIAL_SPLITS = ["training", "validation", "testing"]

# v2 components (essential ones for perception + planning)
V2_COMPONENTS_ESSENTIAL = [
    "lidar",
    "camera_image",
    "lidar_calibration",
    "camera_calibration",
    "vehicle_pose",
    "lidar_box",
    "camera_box",
]

V2_COMPONENTS_ALL = V2_COMPONENTS_ESSENTIAL + [
    "lidar_segmentation",
    "camera_segmentation",
    "camera_hkp",
    "lidar_hkp",
    "lidar_pose",
    "lidar_camera_projection",
    "lidar_camera_synced_box",
    "camera_to_lidar_box_association",
    "projected_lidar_box",
    "stats",
]


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def check_gsutil() -> bool:
    """Check if gsutil is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gsutil", "version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def list_gcs_files(bucket_path: str, pattern: str = "") -> List[str]:
    """List files in a GCS bucket path."""
    full_path = f"{bucket_path}{pattern}" if pattern else bucket_path
    cmd = ["gsutil", "ls", full_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Log the error for debugging
        logger = logging.getLogger(__name__)
        if result.stderr:
            logger.debug(f"gsutil error: {result.stderr.strip()}")
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def get_sequence_ids(bucket: str, split: str, version: str = "v2") -> List[str]:
    """Get unique sequence IDs from a Waymo dataset split."""
    logger = logging.getLogger(__name__)
    
    if version == "v2":
        # For v2, list parquet files in lidar folder
        path = f"gs://{bucket}/{split}/lidar/"
        logger.debug(f"Listing: {path}")
        
        files = list_gcs_files(path)
        logger.debug(f"Found {len(files)} items in lidar/")
        
        if not files:
            # Check if we can access the bucket at all
            test_path = f"gs://{bucket}/"
            test_files = list_gcs_files(test_path)
            if not test_files:
                logger.warning(f"Cannot access bucket gs://{bucket}/ - check authentication")
            else:
                logger.debug(f"Bucket accessible, found: {test_files[:3]}...")
            return []
        
        # Extract sequence IDs from filenames
        sequence_ids = set()
        for f in files:
            name = Path(f).name.rstrip("/")
            if name.endswith(".parquet"):
                seq_id = name.replace(".parquet", "")
                sequence_ids.add(seq_id)
            elif name:
                # Could be a folder name
                sequence_ids.add(name)
        
        return sorted(list(sequence_ids))
    else:
        path = f"gs://{bucket}/{split}/"
        files = list_gcs_files(path, "*.tfrecord")
        return sorted([Path(f).stem for f in files])


def download_sequences(
    sequence_ids: List[str],
    output_dir: Path,
    split: str,
    bucket: str,
    components: List[str],
    version: str,
    logger: logging.Logger,
) -> int:
    """Download specific sequences for a split."""
    downloaded = 0
    dst_split = output_dir / split
    
    for i, seq_id in enumerate(sequence_ids):
        logger.info(f"  [{i+1}/{len(sequence_ids)}] {seq_id}")
        
        if version == "v2":
            for component in components:
                src_path = f"gs://{bucket}/{split}/{component}/"
                dst_path = dst_split / component
                dst_path.mkdir(parents=True, exist_ok=True)
                
                # Find files matching this sequence
                files = list_gcs_files(src_path, f"*{seq_id}*")
                
                for file_url in files:
                    cmd = ["gsutil", "-q", "cp", file_url, str(dst_path)]
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                        downloaded += 1
                    except subprocess.CalledProcessError:
                        pass
        else:
            src_path = f"gs://{bucket}/{split}/"
            dst_split.mkdir(parents=True, exist_ok=True)
            
            files = list_gcs_files(src_path, f"*{seq_id}*")
            for file_url in files:
                cmd = ["gsutil", "-q", "cp", file_url, str(dst_split)]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    downloaded += 1
                except subprocess.CalledProcessError:
                    pass
    
    return downloaded


def download_full_split(
    output_dir: Path,
    split: str,
    bucket: str,
    components: List[str],
    version: str,
    logger: logging.Logger,
):
    """Download an entire official split."""
    logger.info(f"Downloading full {split} split...")
    
    if version == "v2":
        for component in components:
            src = f"gs://{bucket}/{split}/{component}/"
            dst = output_dir / split / component
            dst.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  {component}...")
            cmd = ["gsutil", "-m", "cp", "-r", src, str(dst.parent)]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed: {e}")
    else:
        src = f"gs://{bucket}/{split}/"
        dst = output_dir / split
        dst.mkdir(parents=True, exist_ok=True)
        
        cmd = ["gsutil", "-m", "cp", "-r", src, str(output_dir)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {e}")


def verify_download(output_dir: Path, version: str, logger: logging.Logger):
    """Verify downloaded data and print summary."""
    logger.info("\n" + "=" * 50)
    logger.info("Download Summary")
    logger.info("=" * 50)
    
    for split in OFFICIAL_SPLITS:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        
        if version == "v2":
            parquet_files = list(split_dir.rglob("*.parquet"))
            components = [d.name for d in split_dir.iterdir() if d.is_dir()]
            logger.info(f"{split}: {len(parquet_files)} files, components: {components}")
        else:
            tfrecord_files = list(split_dir.glob("*.tfrecord*"))
            logger.info(f"{split}: {len(tfrecord_files)} tfrecord files")


def main():
    parser = argparse.ArgumentParser(
        description="Download Waymo Open Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 500 train, 150 val, 50 test sequences (medium preset)
  python scripts/download_waymo.py --output-dir data/waymo --subset medium
  
  # Custom counts per split
  python scripts/download_waymo.py --output-dir data/waymo \\
      --num-train 500 --num-val 200 --num-test 100
  
  # Download full training split only
  python scripts/download_waymo.py --output-dir data/waymo --subset full --splits training
  
  # Download all official splits (full dataset)
  python scripts/download_waymo.py --output-dir data/waymo --subset full

Presets:
  tiny   : 30 train, 15 val, 5 test   (~5GB)
  small  : 70 train, 20 val, 10 test  (~10GB)
  medium : 500 train, 150 val, 50 test (~70GB)
  large  : 700 train, 150 val, 50 test (~100GB)
  full   : All sequences              (~1TB)

Note: Requires gsutil with Waymo dataset access. 
      Register at https://waymo.com/open/terms/
        """,
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="medium",
        choices=list(PRESETS.keys()),
        help="Preset subset size (default: medium)",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=None,
        help="Number of training sequences (overrides preset)",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Number of validation sequences (overrides preset)",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=None,
        help="Number of testing sequences (overrides preset)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=OFFICIAL_SPLITS,
        choices=OFFICIAL_SPLITS,
        help="Which splits to download (default: all)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Dataset version: v2=Parquet (recommended), v1=TFRecord",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=None,
        help="Components to download (v2 only)",
    )
    parser.add_argument(
        "--all-components",
        action="store_true",
        help="Download all components including segmentation (v2 only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subset selection (default: 42)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify download after completion",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # Check gsutil
    if not check_gsutil():
        logger.error(
            "gsutil not found or not authenticated.\n"
            "Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install\n"
            "Then authenticate: gcloud auth login"
        )
        sys.exit(1)
    
    # Determine sequence counts per split
    preset = PRESETS[args.subset]
    split_counts = {
        "training": args.num_train if args.num_train is not None else preset["training"],
        "validation": args.num_val if args.num_val is not None else preset["validation"],
        "testing": args.num_test if args.num_test is not None else preset["testing"],
    }
    
    # Filter to requested splits
    split_counts = {k: v for k, v in split_counts.items() if k in args.splits}
    
    # Select components
    if args.components:
        components = args.components
    elif args.all_components:
        components = V2_COMPONENTS_ALL
    else:
        components = V2_COMPONENTS_ESSENTIAL
    
    # Select bucket
    bucket = WAYMO_V2_BUCKET if args.version == "v2" else WAYMO_V1_BUCKET
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("Waymo Open Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"Version: {args.version}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Preset: {args.subset}")
    
    is_full_download = all(v is None for v in split_counts.values())
    
    if is_full_download:
        logger.info("Mode: Full dataset download")
        for split in args.splits:
            logger.info(f"  {split}: all sequences")
    else:
        logger.info("Mode: Subset download")
        for split, count in split_counts.items():
            if count is not None:
                logger.info(f"  {split}: {count} sequences")
            else:
                logger.info(f"  {split}: all sequences")
    
    if args.version == "v2":
        logger.info(f"Components: {components}")
    logger.info("=" * 60)
    
    # Download each split
    errors_occurred = False
    for split in args.splits:
        count = split_counts.get(split)
        
        if count is None:
            # Full download for this split
            download_full_split(output_dir, split, bucket, components, args.version, logger)
        else:
            # Subset download
            logger.info(f"\nFetching available sequences for {split}...")
            available = get_sequence_ids(bucket, split, args.version)
            
            if not available:
                errors_occurred = True
                logger.error(f"Failed to retrieve sequences for {split}.")
                logger.error("  Possible causes:")
                logger.error("    1. Not authenticated: run 'gcloud auth login'")
                logger.error("    2. No Waymo access: accept license at https://waymo.com/open/terms/")
                logger.error("    3. In Colab: run 'from google.colab import auth; auth.authenticate_user()'")
                logger.error("  Use --verbose flag for debug info")
                continue
            
            logger.info(f"  Found {len(available)} sequences available")
            
            # Select random subset
            n_select = min(count, len(available))
            random.seed(args.seed)
            selected = random.sample(available, n_select)
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Downloading {split} ({n_select} sequences)")
            logger.info(f"{'='*50}")
            
            downloaded = download_sequences(
                selected, output_dir, split, bucket, components, args.version, logger
            )
            logger.info(f"Completed {split}: {downloaded} files")
    
    if errors_occurred:
        logger.warning("\nSome splits failed to download. See errors above.")
            logger.info(f"Completed {split}: {downloaded} files")
    
    # Save download info
    info_file = output_dir / "download_info.txt"
    with open(info_file, "w") as f:
        f.write(f"# Waymo Dataset Download\n")
        f.write(f"# Preset: {args.subset}, Seed: {args.seed}\n")
        f.write(f"# Version: {args.version}\n\n")
        for split in args.splits:
            count = split_counts.get(split)
            f.write(f"{split}: {count if count else 'full'}\n")
    
    # Verify
    if args.verify:
        verify_download(output_dir, args.version, logger)
    
    logger.info("\n" + "=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Build cache: python scripts/build_cache.py --input-dir data/waymo --output-dir data/cache")
    logger.info("  2. Train model: python scripts/train.py --config configs/train_tiny.yaml")


if __name__ == "__main__":
    main()
