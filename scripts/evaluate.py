#!/usr/bin/env python3
"""
Evaluation script for Fusion2Drive model.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/eval.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wod_fusion.data import WaymoDataModule
from wod_fusion.models import FusionModel, FusionModelConfig
from wod_fusion.eval import Evaluator


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{output_dir}/eval.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fusion2Drive model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use",
    )
    args = parser.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    logger.info("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Build model
    model_config = checkpoint.get("config", config.get("model", {}))
    
    if isinstance(model_config, dict):
        model_config = FusionModelConfig(**model_config)
    
    model = FusionModel(model_config)
    model.load_state_dict(checkpoint["model"])
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Build datamodule
    data_config = config["data"]
    datamodule = WaymoDataModule(
        data_dir=data_config["data_dir"],
        cache_dir=data_config.get("cache_dir"),
        batch_size=data_config.get("batch_size", 1),
        num_workers=data_config.get("num_workers", 4),
        image_size=tuple(data_config.get("image_size", [256, 704])),
        max_points=data_config.get("max_points", 150000),
        augment=False,  # No augmentation for eval
    )
    datamodule.setup()
    
    # Get dataloader
    if args.split == "validation":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Run evaluation
    eval_config = config.get("eval", {})
    
    if args.ablation:
        logger.info("Running ablation study...")
        results = evaluator.run_ablation(
            score_threshold=eval_config.get("score_threshold", 0.3),
        )
        
        logger.info("\nAblation Results:")
        for name, metrics in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  mAP@0.7: {metrics['mAP']:.4f}")
            logger.info(f"  ADE: {metrics['ADE']:.4f} m")
            logger.info(f"  FDE: {metrics['FDE']:.4f} m")
    else:
        metrics = evaluator.evaluate(
            score_threshold=eval_config.get("score_threshold", 0.3),
            nms_threshold=eval_config.get("nms_threshold", 0.5),
            compute_runtime=True,
        )
        
        logger.info(f"\nResults saved to {args.output_dir}")
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
