#!/usr/bin/env python3
"""
Export model to ONNX, CoreML, and TorchScript.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best.pt --format all
    python scripts/export_model.py --checkpoint checkpoints/best.pt --format coreml --optimize
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wod_fusion.models import FusionModel, FusionModelConfig
from wod_fusion.export import ModelExporter, ExportConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(description="Export Fusion2Drive model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exports",
        help="Output directory for exported models",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["all", "onnx", "torchscript", "coreml"],
        help="Export format",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "lidar_only", "camera_only"],
        help="Model mode to export",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize for mobile deployment",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for export",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark exported model",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    model_config = checkpoint.get("config", {})
    if isinstance(model_config, dict):
        model_config = FusionModelConfig(**model_config)
    
    model = FusionModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Export config
    export_config = ExportConfig(
        batch_size=args.batch_size,
        optimize_for_mobile=args.optimize,
    )
    
    # Create exporter
    exporter = ModelExporter(
        model=model,
        output_dir=args.output_dir,
        config=export_config,
    )
    
    # Determine formats
    if args.format == "all":
        formats = ["onnx", "torchscript", "coreml"]
    else:
        formats = [args.format]
    
    # Export
    logger.info(f"Exporting to: {formats}")
    results = exporter.export_all(mode=args.mode, formats=formats)
    
    for format_name, path in results.items():
        logger.info(f"Exported {format_name}: {path}")
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("\nBenchmarking exported models...")
        
        for format_name, path in results.items():
            try:
                stats = exporter.benchmark(format_name, path)
                logger.info(f"\n{format_name.upper()} Performance:")
                logger.info(f"  Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
                logger.info(f"  FPS: {stats['fps']:.1f}")
            except Exception as e:
                logger.warning(f"Could not benchmark {format_name}: {e}")
    
    logger.info("\nExport complete!")


if __name__ == "__main__":
    main()
