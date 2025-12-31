#!/usr/bin/env python3
"""
Download Waymo Open Dataset.

This script helps download the Waymo Open Dataset v2.0 (Parquet format)
or v1.4.3 (TFRecord format) from Google Cloud Storage.

Usage:
    # Download v2.0 Parquet format (recommended)
    python scripts/download_waymo.py --version v2 --output_dir /data/waymo
    
    # Download v1.4.3 TFRecord format
    python scripts/download_waymo.py --version v1 --output_dir /data/waymo
    
    # Download only validation split
    python scripts/download_waymo.py --version v2 --split validation --output_dir /data/waymo
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

try:
    from google.cloud import storage
except ImportError:
    storage = None


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# GCS bucket paths
WAYMO_V2_BUCKET = "waymo_open_dataset_v_2_0_1"
WAYMO_V1_BUCKET = "waymo_open_dataset_v_1_4_3"

# v2 components
V2_COMPONENTS = [
    "lidar",
    "camera_image",
    "lidar_calibration",
    "camera_calibration",
    "vehicle_pose",
    "lidar_box",
    "camera_box",
    "lidar_segmentation",
    "camera_segmentation",
]


def check_gsutil():
    """Check if gsutil is installed."""
    try:
        result = subprocess.run(
            ["gsutil", "version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_v2(output_dir: str, split: str = "all", components: list = None):
    """Download Waymo v2.0 Parquet dataset using gsutil."""
    logger = logging.getLogger(__name__)
    
    if not check_gsutil():
        logger.error(
            "gsutil not found. Please install Google Cloud SDK:\n"
            "https://cloud.google.com/sdk/docs/install"
        )
        sys.exit(1)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["training", "validation", "testing"] if split == "all" else [split]
    components = components or V2_COMPONENTS
    
    for s in splits:
        for component in components:
            src = f"gs://{WAYMO_V2_BUCKET}/{s}/{component}/"
            dst = output_dir / s / component
            dst.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {s}/{component}...")
            
            cmd = [
                "gsutil", "-m", "cp", "-r",
                src,
                str(dst.parent),
            ]
            
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"Completed {s}/{component}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download {s}/{component}: {e}")
    
    logger.info(f"Download complete. Data saved to {output_dir}")


def download_v1(output_dir: str, split: str = "all"):
    """Download Waymo v1.4.3 TFRecord dataset using gsutil."""
    logger = logging.getLogger(__name__)
    
    if not check_gsutil():
        logger.error(
            "gsutil not found. Please install Google Cloud SDK:\n"
            "https://cloud.google.com/sdk/docs/install"
        )
        sys.exit(1)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["training", "validation", "testing"] if split == "all" else [split]
    
    for s in splits:
        src = f"gs://{WAYMO_V1_BUCKET}/{s}/"
        dst = output_dir / s
        dst.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {s}...")
        
        cmd = [
            "gsutil", "-m", "cp", "-r",
            src,
            str(output_dir),
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Completed {s}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {s}: {e}")
    
    logger.info(f"Download complete. Data saved to {output_dir}")


def verify_download(output_dir: str, version: str):
    """Verify downloaded data."""
    logger = logging.getLogger(__name__)
    
    output_dir = Path(output_dir)
    
    if version == "v2":
        # Check for Parquet files
        parquet_files = list(output_dir.rglob("*.parquet"))
        logger.info(f"Found {len(parquet_files)} Parquet files")
        
        # Check components
        for split in ["training", "validation"]:
            split_dir = output_dir / split
            if split_dir.exists():
                components = [d.name for d in split_dir.iterdir() if d.is_dir()]
                logger.info(f"{split}: {components}")
    else:
        # Check for TFRecord files
        tfrecord_files = list(output_dir.rglob("*.tfrecord*"))
        logger.info(f"Found {len(tfrecord_files)} TFRecord files")
        
        for split in ["training", "validation"]:
            split_dir = output_dir / split
            if split_dir.exists():
                num_files = len(list(split_dir.glob("*.tfrecord*")))
                logger.info(f"{split}: {num_files} files")


def main():
    parser = argparse.ArgumentParser(description="Download Waymo Open Dataset")
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Dataset version (v2 = Parquet, v1 = TFRecord)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "training", "validation", "testing"],
        help="Split to download",
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        default=None,
        help="Components to download (v2 only)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify download after completion",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Downloading Waymo Open Dataset {args.version}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if args.version == "v2":
        download_v2(args.output_dir, args.split, args.components)
    else:
        download_v1(args.output_dir, args.split)
    
    if args.verify:
        logger.info("\nVerifying download...")
        verify_download(args.output_dir, args.version)
    
    logger.info("\nDone!")
    logger.info(
        "\nNote: You need to accept the Waymo Open Dataset license at:\n"
        "https://waymo.com/open/terms/"
    )


if __name__ == "__main__":
    main()
