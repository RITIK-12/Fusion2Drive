"""
Checkpoint utilities for model saving and loading.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Any, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: Optional[str] = None,
    config: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    keep_last_n: int = 5,
    is_best: bool = False,
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dict of metrics to save
        checkpoint_dir: Directory to save checkpoint
        filename: Optional specific filename
        config: Model/training config to save
        scheduler: LR scheduler state
        scaler: GradScaler for mixed precision
        keep_last_n: Number of recent checkpoints to keep
        is_best: Whether this is the best checkpoint
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle DataParallel/DistributedDataParallel
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Build checkpoint dict
    checkpoint = {
        "epoch": epoch,
        "model": model_state,
        "metrics": metrics,
    }
    
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    
    if config is not None:
        # Convert dataclass to dict if needed
        if hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            checkpoint["config"] = asdict(config)
        else:
            checkpoint["config"] = config
    
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    
    # Determine filename
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
    
    filepath = checkpoint_dir / filename
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint: {filepath}")
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint: {best_path}")
    
    # Cleanup old checkpoints
    if keep_last_n > 0:
        _cleanup_checkpoints(checkpoint_dir, keep_last_n)
    
    return str(filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        scaler: GradScaler to load state into
        map_location: Device to map tensors to
        strict: Whether to enforce strict state dict loading
        
    Returns:
        Checkpoint dict with epoch, metrics, config
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Load model state
    if model is not None and "model" in checkpoint:
        # Handle DataParallel/DistributedDataParallel
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model"], strict=strict)
        else:
            model.load_state_dict(checkpoint["model"], strict=strict)
        logger.info("Loaded model state")
    
    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Loaded optimizer state")
    
    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("Loaded scheduler state")
    
    # Load scaler state
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
        logger.info("Loaded scaler state")
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }


def find_best_checkpoint(
    checkpoint_dir: str,
    metric_name: str = "val_loss",
    mode: str = "min",
) -> Optional[str]:
    """
    Find the best checkpoint based on a metric.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric_name: Name of metric to compare
        mode: "min" or "max"
        
    Returns:
        Path to best checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Check for explicit best checkpoint
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return str(best_path)
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if not checkpoints:
        return None
    
    best_checkpoint = None
    best_value = float("inf") if mode == "min" else float("-inf")
    
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            metrics = ckpt.get("metrics", {})
            
            if metric_name in metrics:
                value = metrics[metric_name]
                
                if mode == "min" and value < best_value:
                    best_value = value
                    best_checkpoint = str(ckpt_path)
                elif mode == "max" and value > best_value:
                    best_value = value
                    best_checkpoint = str(ckpt_path)
        except Exception as e:
            logger.warning(f"Could not load checkpoint {ckpt_path}: {e}")
    
    return best_checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint by epoch number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if not checkpoints:
        return None
    
    # Extract epoch numbers and sort
    def get_epoch(path: Path) -> int:
        match = re.search(r"epoch_(\d+)", path.name)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_epoch)
    
    return str(checkpoints[-1])


def _cleanup_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """
    Remove old checkpoints, keeping only the last N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
    """
    # Find all epoch checkpoints
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by epoch
    def get_epoch(path: Path) -> int:
        match = re.search(r"epoch_(\d+)", path.name)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_epoch)
    
    # Remove old checkpoints
    for ckpt in checkpoints[:-keep_last_n]:
        ckpt.unlink()
        logger.debug(f"Removed old checkpoint: {ckpt}")


def average_checkpoints(
    checkpoint_paths: list,
    model: nn.Module,
    output_path: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Average weights from multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths to average
        model: Model architecture (for state dict structure)
        output_path: Optional path to save averaged weights
        
    Returns:
        Averaged state dict
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")
    
    logger.info(f"Averaging {len(checkpoint_paths)} checkpoints")
    
    # Load first checkpoint
    avg_state = torch.load(checkpoint_paths[0], map_location="cpu")["model"]
    
    # Convert to float for averaging
    for key in avg_state:
        avg_state[key] = avg_state[key].float()
    
    # Add remaining checkpoints
    for path in checkpoint_paths[1:]:
        state = torch.load(path, map_location="cpu")["model"]
        for key in avg_state:
            avg_state[key] += state[key].float()
    
    # Average
    for key in avg_state:
        avg_state[key] /= len(checkpoint_paths)
    
    # Save if output path provided
    if output_path:
        torch.save({"model": avg_state}, output_path)
        logger.info(f"Saved averaged checkpoint: {output_path}")
    
    return avg_state


def convert_to_half_precision(
    checkpoint_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Convert checkpoint to half precision (float16).
    
    Args:
        checkpoint_path: Path to checkpoint
        output_path: Output path (default: adds _fp16 suffix)
        
    Returns:
        Path to converted checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Convert model weights to half precision
    for key in checkpoint["model"]:
        if checkpoint["model"][key].dtype == torch.float32:
            checkpoint["model"][key] = checkpoint["model"][key].half()
    
    # Determine output path
    if output_path is None:
        path = Path(checkpoint_path)
        output_path = str(path.parent / f"{path.stem}_fp16{path.suffix}")
    
    torch.save(checkpoint, output_path)
    
    # Log size reduction
    original_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    new_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    logger.info(f"Converted to FP16: {original_size:.2f} MB -> {new_size:.2f} MB")
    
    return output_path
