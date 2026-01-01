"""
Data module for PyTorch DataLoader management.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader

from wod_fusion.data.dataset import WaymoDataset, SensorConfig

logger = logging.getLogger(__name__)


class WaymoDataModule:
    """
    Data module for Waymo Open Dataset.
    
    Handles:
    - Train/val/test dataset creation
    - DataLoader configuration
    - Worker and memory management
    """
    
    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        sensor_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        max_train_segments: Optional[int] = None,
        max_val_segments: Optional[int] = None,
    ):
        """
        Initialize data module.
        
        Args:
            data_dir: Path to Waymo dataset root
            cache_dir: Path to cache directory
            sensor_config: Sensor configuration dict
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            max_train_segments: Maximum training segments (for debugging)
            max_val_segments: Maximum validation segments
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_train_segments = max_train_segments
        self.max_val_segments = max_val_segments
        
        # Build sensor config
        self.sensor_config = SensorConfig(
            **(sensor_config or {})
        )
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, or testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or None (all)
        """
        if stage in (None, "fit"):
            self._train_dataset = WaymoDataset(
                data_dir=str(self.data_dir),
                split="training",
                config=self.sensor_config,
                cache_dir=str(self.cache_dir / "training"),
                use_cache=True,
                transform=True,
                max_segments=self.max_train_segments,
            )
            logger.info(f"Training dataset: {len(self._train_dataset)} frames")
        
        if stage in (None, "fit", "validate"):
            self._val_dataset = WaymoDataset(
                data_dir=str(self.data_dir),
                split="validation",
                config=self.sensor_config,
                cache_dir=str(self.cache_dir / "validation"),
                use_cache=True,
                transform=True,
                max_segments=self.max_val_segments,
            )
            logger.info(f"Validation dataset: {len(self._val_dataset)} frames")
        
        if stage in (None, "test"):
            self._test_dataset = WaymoDataset(
                data_dir=str(self.data_dir),
                split="testing",
                config=self.sensor_config,
                cache_dir=str(self.cache_dir / "testing"),
                use_cache=True,
                transform=True,
            )
            logger.info(f"Test dataset: {len(self._test_dataset)} frames")
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        if self._train_dataset is None:
            self.setup(stage="fit")
        
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._train_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self._val_dataset is None:
            self.setup(stage="validate")
        
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._val_dataset.collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        if self._test_dataset is None:
            self.setup(stage="test")
        
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._test_dataset.collate_fn,
            drop_last=False,
        )
    
    @property
    def train_dataset(self) -> WaymoDataset:
        if self._train_dataset is None:
            self.setup(stage="fit")
        return self._train_dataset
    
    @property
    def val_dataset(self) -> WaymoDataset:
        if self._val_dataset is None:
            self.setup(stage="validate")
        return self._val_dataset
