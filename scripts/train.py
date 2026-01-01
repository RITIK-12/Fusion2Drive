#!/usr/bin/env python3
"""
Training script for Fusion2Drive model.

Usage:
    python scripts/train.py --config configs/train_full.yaml
    python scripts/train.py --config configs/train_tiny.yaml --debug
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
from wod_fusion.training import Trainer, TrainerConfig


def setup_logging(log_dir: str, debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_dir}/train.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def build_model(model_config: dict) -> FusionModel:
    """Build model from configuration."""
    config = FusionModelConfig(
        use_lidar=model_config.get("use_lidar", True),
        use_camera=model_config.get("use_camera", True),
        lidar_hidden_channels=model_config.get("lidar_channels", 64),
        lidar_out_channels=model_config.get("lidar_channels", 128),
        camera_out_channels=model_config.get("camera_channels", 128),
        camera_backbone=model_config.get("camera_backbone", "resnet18"),
        bev_in_channels=model_config.get("bev_channels", 256),
        bev_hidden_channels=model_config.get("bev_channels", 256),
        bev_out_channels=model_config.get("bev_channels", 256),
        detection_num_classes=model_config.get("num_classes", 3),
        planning_num_waypoints=model_config.get("num_waypoints", 12),
    )
    
    return FusionModel(config)


def build_datamodule(data_config: dict) -> WaymoDataModule:
    """Build data module from configuration."""
    # Build sensor config from data config
    sensor_config = {
        "image_size": tuple(data_config.get("image_size", [640, 480])),
        "max_points": data_config.get("max_points", 150000),
    }
    
    return WaymoDataModule(
        data_dir=data_config["data_dir"],
        cache_dir=data_config.get("cache_dir"),
        sensor_config=sensor_config,
        batch_size=data_config.get("batch_size", 4),
        num_workers=data_config.get("num_workers", 4),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Fusion2Drive model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU devices to use (comma-separated)",
    )
    args = parser.parse_args()
    
    # Set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_dir = config.get("log_dir", "outputs/train")
    setup_logging(log_dir, args.debug)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with config: {args.config}")
    
    # Log config
    logger.info(f"Configuration:\n{yaml.dump(config, default_flow_style=False)}")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config.get("model", {}))
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Build data module
    logger.info("Building data module...")
    datamodule = build_datamodule(config["data"])
    datamodule.setup()
    
    # Build trainer
    trainer_config = config.get("trainer", {})
    loss_config = config.get("loss", {})
    trainer = Trainer(
        config=TrainerConfig(
            max_epochs=trainer_config.get("max_epochs", 50),
            learning_rate=trainer_config.get("learning_rate", 1e-4),
            weight_decay=trainer_config.get("weight_decay", 0.01),
            grad_accum_steps=trainer_config.get("gradient_accumulation_steps", 1),
            use_amp=trainer_config.get("mixed_precision", True),
            warmup_epochs=trainer_config.get("warmup_epochs", 5),
            log_every_n_steps=trainer_config.get("log_interval", 100),
            save_every_n_epochs=trainer_config.get("save_interval", 1),
            checkpoint_dir=trainer_config.get("checkpoint_dir", "checkpoints"),
            log_dir=log_dir,
            detection_weight=loss_config.get("detection_weight", 1.0),
            planning_weight=loss_config.get("planning_weight", 1.0),
        ),
        model=model,
        train_loader=datamodule.train_dataloader(),
        val_loader=datamodule.val_dataloader(),
        device=device,
    )
    
    # Train (with optional resume)
    logger.info("Starting training...")
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
    trainer.train(resume_from=args.resume)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
