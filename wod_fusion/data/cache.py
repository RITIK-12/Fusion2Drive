"""
Cache builder for preprocessing Waymo dataset.

Preprocesses raw data and saves to disk for fast training.
Supports resuming from interruption.
"""

import os
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from wod_fusion.data.dataset import WaymoDataset, SensorConfig
from wod_fusion.utils.geometry import get_future_waypoints_from_poses

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache building."""
    # Data settings
    cameras: List[str] = None
    image_size: tuple = (640, 480)
    point_cloud_range: List[float] = None
    max_points: int = 150000
    
    # Waypoint settings
    num_future_frames: int = 10
    frame_interval: float = 0.1  # 10Hz
    
    # Processing settings
    num_workers: int = 8
    chunk_size: int = 100
    
    def __post_init__(self):
        if self.cameras is None:
            self.cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
        if self.point_cloud_range is None:
            self.point_cloud_range = [-75.0, -75.0, -3.0, 75.0, 75.0, 5.0]


class CacheBuilder:
    """
    Build cache from Waymo Open Dataset.
    
    Processes:
    1. Camera images (resized, normalized)
    2. LiDAR point clouds (filtered, downsampled)
    3. Calibration data
    4. 3D labels
    5. Future ego waypoints
    
    Saves per-frame .pt files for fast loading during training.
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize cache builder.
        
        Args:
            input_dir: Path to Waymo dataset split directory
            output_dir: Path to output cache directory
            config: Cache configuration
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or CacheConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track progress for resume
        self.progress_file = self.output_dir / "progress.txt"
        self.completed = self._load_progress()
    
    def _load_progress(self) -> set:
        """Load completed frame IDs from progress file."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                return set(line.strip() for line in f)
        return set()
    
    def _save_progress(self, frame_id: str):
        """Save completed frame ID to progress file."""
        with open(self.progress_file, "a") as f:
            f.write(f"{frame_id}\n")
        self.completed.add(frame_id)
    
    def build(self, max_segments: Optional[int] = None):
        """
        Build cache for all frames.
        
        Args:
            max_segments: Maximum number of segments to process
        """
        # Create dataset without cache to get frame index
        sensor_config = SensorConfig(
            cameras=self.config.cameras,
            image_size=self.config.image_size,
            point_cloud_range=self.config.point_cloud_range,
            max_points=self.config.max_points,
            num_future_frames=self.config.num_future_frames,
            frame_interval=self.config.frame_interval,
        )
        
        dataset = WaymoDataset(
            data_dir=str(self.input_dir.parent),
            split=self.input_dir.name,
            config=sensor_config,
            use_cache=False,
            transform=False,
            max_segments=max_segments,
            return_raw=True,
        )
        
        logger.info(f"Building cache for {len(dataset)} frames...")
        
        # Process frames
        pending = []
        for idx in range(len(dataset)):
            frame_info = dataset.frame_index[idx]
            frame_id = f"{frame_info['segment_id']}_{frame_info.get('timestamp', frame_info['frame_idx'])}"
            
            if frame_id not in self.completed:
                pending.append((idx, frame_id, frame_info))
        
        logger.info(f"Processing {len(pending)} frames ({len(self.completed)} already cached)")
        
        if self.config.num_workers > 1:
            self._process_parallel(dataset, pending)
        else:
            self._process_sequential(dataset, pending)
        
        # Save index
        self._save_index(dataset)
        
        logger.info(f"Cache building complete: {len(dataset)} frames")
    
    def _process_sequential(
        self,
        dataset: WaymoDataset,
        pending: List[tuple],
    ):
        """Process frames sequentially."""
        for idx, frame_id, frame_info in tqdm(pending, desc="Building cache"):
            try:
                # Load raw data
                data = dataset[idx]
                
                # Save to cache
                cache_path = self.output_dir / f"{frame_id}.pt"
                self._save_frame(data, cache_path)
                
                self._save_progress(frame_id)
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_id}: {e}")
                continue
    
    def _process_parallel(
        self,
        dataset: WaymoDataset,
        pending: List[tuple],
    ):
        """Process frames in parallel."""
        # Note: For TFRecord format, parallel processing is tricky
        # because TFRecordDataset doesn't support random access well.
        # For Parquet format, this works well.
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {}
            
            for idx, frame_id, frame_info in pending:
                future = executor.submit(
                    _process_frame_worker,
                    str(self.input_dir),
                    frame_info,
                    str(self.output_dir / f"{frame_id}.pt"),
                    self.config,
                )
                futures[future] = frame_id
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building cache"):
                frame_id = futures[future]
                try:
                    future.result()
                    self._save_progress(frame_id)
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_id}: {e}")
    
    def _save_frame(self, data: Dict, path: Path):
        """Save frame data to disk."""
        # Convert numpy arrays to tensors
        save_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                save_data[key] = torch.from_numpy(value)
            elif isinstance(value, torch.Tensor):
                save_data[key] = value
            else:
                save_data[key] = value
        
        torch.save(save_data, path)
    
    def _save_index(self, dataset: WaymoDataset):
        """Save frame index for fast loading."""
        index = []
        for frame_info in dataset.frame_index:
            frame_id = f"{frame_info['segment_id']}_{frame_info.get('timestamp', frame_info['frame_idx'])}"
            index.append({
                **frame_info,
                "cache_path": str(self.output_dir / f"{frame_id}.pt"),
            })
        
        split = self.input_dir.name
        torch.save(index, self.output_dir / f"{split}_index.pt")
    
    def verify(self) -> Dict[str, Any]:
        """
        Verify cache integrity.
        
        Returns:
            Dict with verification results
        """
        cache_files = list(self.output_dir.glob("*.pt"))
        
        valid = 0
        invalid = []
        
        for cache_file in tqdm(cache_files, desc="Verifying cache"):
            try:
                data = torch.load(cache_file)
                
                # Check required keys
                required_keys = ["images", "points", "boxes_3d", "waypoints"]
                for key in required_keys:
                    if key not in data:
                        raise ValueError(f"Missing key: {key}")
                
                valid += 1
            except Exception as e:
                invalid.append((cache_file.name, str(e)))
        
        return {
            "total": len(cache_files),
            "valid": valid,
            "invalid": len(invalid),
            "invalid_files": invalid[:10],  # First 10 invalid files
        }


def _process_frame_worker(
    input_dir: str,
    frame_info: Dict,
    output_path: str,
    config: CacheConfig,
) -> bool:
    """Worker function for parallel cache building."""
    from wod_fusion.data.dataset import WaymoDataset, SensorConfig
    
    sensor_config = SensorConfig(
        cameras=config.cameras,
        image_size=config.image_size,
        point_cloud_range=config.point_cloud_range,
        max_points=config.max_points,
        num_future_frames=config.num_future_frames,
        frame_interval=config.frame_interval,
    )
    
    # Create minimal dataset for single frame loading
    dataset = WaymoDataset(
        data_dir=str(Path(input_dir).parent),
        split=Path(input_dir).name,
        config=sensor_config,
        use_cache=False,
        transform=False,
        return_raw=True,
    )
    
    # Find frame index
    for idx, fi in enumerate(dataset.frame_index):
        if fi["segment_id"] == frame_info["segment_id"] and fi.get("timestamp") == frame_info.get("timestamp"):
            data = dataset[idx]
            
            # Save
            save_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    save_data[key] = torch.from_numpy(value)
                elif isinstance(value, torch.Tensor):
                    save_data[key] = value
                else:
                    save_data[key] = value
            
            torch.save(save_data, output_path)
            return True
    
    return False
