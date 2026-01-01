"""
Training loop with mixed precision and gradient accumulation.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from wod_fusion.models import FusionModel
from wod_fusion.training.losses import MultiTaskLoss, create_detection_targets
from wod_fusion.training.scheduler import WarmupCosineScheduler
from wod_fusion.utils.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_epochs: int = 24
    warmup_epochs: int = 2
    
    # Batch settings
    batch_size: int = 8
    grad_accum_steps: int = 4
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "outputs/checkpoints"
    save_every_n_epochs: int = 1
    
    # Logging
    log_dir: str = "outputs/logs"
    log_every_n_steps: int = 50
    
    # Validation
    val_every_n_epochs: int = 1
    
    # Early stopping
    patience: int = 5
    
    # Loss weights
    detection_weight: float = 1.0
    planning_weight: float = 2.0
    
    # BEV settings for target creation
    bev_x_range: tuple = (-75.0, 75.0)
    bev_y_range: tuple = (-75.0, 75.0)
    bev_resolution: float = 0.5
    
    # Seed
    seed: int = 42
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "fusion2drive"
    wandb_run_name: Optional[str] = None


class Trainer:
    """
    Training loop for Fusion2Drive model.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Warmup + cosine learning rate schedule
    - TensorBoard logging
    - Checkpointing with resume support
    - Early stopping
    - Multi-GPU support (via DataParallel)
    """
    
    def __init__(
        self,
        model: FusionModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: FusionModel to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on ('cuda', 'mps', 'cpu')
        """
        self.config = config or TrainerConfig()
        
        # Set seed
        torch.manual_seed(self.config.seed)
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        logger.info(f"Training on device: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Multi-GPU
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function
        self.criterion = MultiTaskLoss(
            detection_weight=self.config.detection_weight,
            planning_weight=self.config.planning_weight,
            use_uncertainty_weighting=False,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler
        steps_per_epoch = len(train_loader) // self.config.grad_accum_steps
        total_steps = steps_per_epoch * self.config.max_epochs
        warmup_steps = steps_per_epoch * self.config.warmup_epochs
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        
        # Mixed precision
        self.scaler = GradScaler(enabled=self.config.use_amp and self.device.type == "cuda")
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        
        # Logging
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Wandb
        self.use_wandb = self.config.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config={
                        "learning_rate": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay,
                        "max_epochs": self.config.max_epochs,
                        "batch_size": self.config.batch_size,
                        "grad_accum_steps": self.config.grad_accum_steps,
                        "use_amp": self.config.use_amp,
                    },
                )
                logger.info(f"Wandb initialized: {wandb.run.name}")
            except ImportError:
                logger.warning("Wandb not installed, disabling")
                self.use_wandb = False
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def train(self, resume_from: Optional[str] = None):
        """
        Run training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        logger.info(f"Starting training from epoch {self.epoch}")
        
        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            
            # Train one epoch
            train_metrics = self._train_epoch()
            
            # Log training metrics
            self._log_metrics(train_metrics, prefix="train")
            
            # Validation
            if self.val_loader is not None and epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self._validate()
                self._log_metrics(val_metrics, prefix="val")
                
                # Check for improvement
                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    self.patience_counter = 0
                    self._save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}.pt")
        
        # Save final checkpoint
        self._save_checkpoint("final.pt")
        
        logger.info("Training complete!")
        self.writer.close()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self._to_device(batch)
            
            # Create targets
            targets = self._create_targets(batch)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    images=batch["images"],
                    intrinsics=batch["intrinsics"],
                    extrinsics=batch["extrinsics"],
                    points=batch["points"],
                    points_mask=batch["points_mask"],
                )
                
                losses = self.criterion(outputs, targets)
                loss = losses["total"] / self.config.grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # Clip gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                self.scheduler.step()
                
                self.global_step += 1
                
                # Log
                if self.global_step % self.config.log_every_n_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    for name, value in losses.items():
                        self.writer.add_scalar(f"train/{name}", value.item(), self.global_step)
            
            # Accumulate losses
            for name, value in losses.items():
                if name not in epoch_losses:
                    epoch_losses[name] = 0.0
                epoch_losses[name] += value.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "det": f"{losses['detection_total'].item():.4f}",
                "plan": f"{losses['waypoint_total'].item():.4f}",
            })
        
        # Average losses
        for name in epoch_losses:
            epoch_losses[name] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        val_losses = {}
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._to_device(batch)
            targets = self._create_targets(batch)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    images=batch["images"],
                    intrinsics=batch["intrinsics"],
                    extrinsics=batch["extrinsics"],
                    points=batch["points"],
                    points_mask=batch["points_mask"],
                )
                
                losses = self.criterion(outputs, targets)
            
            for name, value in losses.items():
                if name not in val_losses:
                    val_losses[name] = 0.0
                val_losses[name] += value.item()
            num_batches += 1
        
        for name in val_losses:
            val_losses[name] /= num_batches
        
        return val_losses
    
    def _create_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create training targets from batch."""
        # Get BEV size from model config
        if hasattr(self.model, "module"):
            config = self.model.module.config
        else:
            config = self.model.config
        
        bev_size = config.bev_size
        
        # Create detection targets
        det_targets = create_detection_targets(
            boxes_3d=batch["boxes_3d"],
            boxes_mask=batch["boxes_mask"],
            heatmap_size=bev_size,
            bev_x_range=self.config.bev_x_range,
            bev_y_range=self.config.bev_y_range,
            bev_resolution=self.config.bev_resolution,
            num_classes=config.detection_num_classes,
        )
        
        return {
            "heatmap": det_targets["heatmap"],
            "box_reg": det_targets["box_reg"],
            "reg_mask": det_targets["reg_mask"],
            "waypoints": batch["waypoints"],
        }
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str):
        """Log metrics to TensorBoard and Wandb."""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}_epoch/{name}", value, self.epoch)
        
        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.epoch)
        
        logger.info(f"Epoch {self.epoch} {prefix}: " + 
                    ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": vars(model_to_save.config),
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
