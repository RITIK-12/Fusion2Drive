"""
Waymo Open Dataset parser and dataset class.

Supports both v2.0.1 (Parquet) and v1.4.3 (TFRecords) formats.
Handles multi-camera images, LiDAR point clouds, calibration, and labels.
"""

import io
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from wod_fusion.data.transforms import ImageTransform, PointCloudTransform
from wod_fusion.data.waymo_parser import WaymoFrameParser
from wod_fusion.utils.geometry import (
    transform_points,
    get_future_waypoints_from_poses,
)

logger = logging.getLogger(__name__)


@dataclass
class SensorConfig:
    """Configuration for sensor data loading."""
    # Camera settings
    cameras: List[str] = None
    image_size: Tuple[int, int] = (640, 480)  # (width, height)
    
    # LiDAR settings
    point_cloud_range: List[float] = None
    max_points: int = 150000
    
    # Temporal settings
    num_past_frames: int = 0
    num_future_frames: int = 10  # For waypoint labels
    frame_interval: float = 0.1  # 10Hz
    
    def __post_init__(self):
        if self.cameras is None:
            # Default: 5 cameras (all except rear)
            self.cameras = [
                "FRONT", 
                "FRONT_LEFT", 
                "FRONT_RIGHT",
                "SIDE_LEFT",
                "SIDE_RIGHT",
            ]
        if self.point_cloud_range is None:
            # Default: 150m x 150m, -3m to 5m height
            self.point_cloud_range = [-75.0, -75.0, -3.0, 75.0, 75.0, 5.0]


class WaymoDataset(Dataset):
    """
    Dataset for Waymo Open Dataset with multi-sensor fusion support.
    
    Loads:
    - Multi-camera images (configurable subset)
    - LiDAR point clouds (merged from all LiDAR sensors)
    - Camera intrinsics and extrinsics
    - Ego vehicle poses for coordinate transformation
    - 3D object labels (vehicles, pedestrians, cyclists)
    - Future ego waypoints derived from ego poses
    
    Data Flow:
    1. Raw data from Waymo (Parquet or TFRecords)
    2. Parsed into per-frame data
    3. Optionally cached to disk for fast loading
    4. Transformed and batched for training
    
    Coordinate Frames:
    - All outputs are in ego vehicle frame at keyframe timestamp
    - LiDAR points transformed from sensor frame to ego frame
    - Camera extrinsics define camera-to-ego transformation
    """
    
    # Waymo camera name mapping
    CAMERA_NAMES = {
        "FRONT": 1,
        "FRONT_LEFT": 2,
        "FRONT_RIGHT": 3,
        "SIDE_LEFT": 4,
        "SIDE_RIGHT": 5,
    }
    
    # Waymo LiDAR name mapping
    LIDAR_NAMES = {
        "TOP": 1,
        "FRONT": 2,
        "SIDE_LEFT": 3,
        "SIDE_RIGHT": 4,
        "REAR": 5,
    }
    
    # Class mapping for detection
    CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
    CLASS_MAP = {
        1: 0,  # TYPE_VEHICLE -> vehicle
        2: 1,  # TYPE_PEDESTRIAN -> pedestrian  
        4: 2,  # TYPE_CYCLIST -> cyclist
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        config: Optional[SensorConfig] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        transform: bool = True,
        max_segments: Optional[int] = None,
        return_raw: bool = False,
    ):
        """
        Initialize Waymo dataset.
        
        Args:
            data_dir: Path to Waymo dataset root
            split: Dataset split ('training', 'validation', 'testing')
            config: Sensor configuration
            cache_dir: Path to cache directory (if None, uses data_dir/cache)
            use_cache: Whether to use cached data if available
            transform: Whether to apply data transforms
            max_segments: Maximum number of segments to load (for debugging)
            return_raw: If True, return raw data without processing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or SensorConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / "cache"
        self.use_cache = use_cache
        self.transform = transform
        self.return_raw = return_raw
        
        # Initialize transforms
        self.image_transform = ImageTransform(
            target_size=self.config.image_size,
            normalize=True,
        )
        self.point_cloud_transform = PointCloudTransform(
            point_cloud_range=self.config.point_cloud_range,
            max_points=self.config.max_points,
        )
        
        # Initialize parser
        self.parser = WaymoFrameParser(data_format=self._detect_format())
        
        # Build index of all frames
        self.frame_index = self._build_frame_index(max_segments)
        
        logger.info(
            f"Initialized WaymoDataset: {len(self.frame_index)} frames, "
            f"{len(self.config.cameras)} cameras, split={split}"
        )
    
    def _detect_format(self) -> str:
        """Detect whether data is in Parquet (v2) or TFRecord (v1) format."""
        parquet_dir = self.data_dir / self.split / "camera_image"
        tfrecord_dir = self.data_dir / self.split
        
        if parquet_dir.exists():
            return "parquet"
        elif any(tfrecord_dir.glob("*.tfrecord*")):
            return "tfrecord"
        else:
            # Check cache
            if self.cache_dir.exists() and any(self.cache_dir.glob("*.pt")):
                return "cache"
            raise ValueError(
                f"Could not detect Waymo data format in {self.data_dir}. "
                "Expected Parquet files in {split}/camera_image/ or TFRecord files."
            )
    
    def _build_frame_index(
        self, 
        max_segments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build index of all frames in the dataset.
        
        Returns list of dicts with:
        - segment_id: Unique segment identifier
        - frame_idx: Frame index within segment
        - cache_path: Path to cached frame data (if exists)
        - raw_paths: Paths to raw data files (if not cached)
        """
        frame_index = []
        
        # Check cache first
        if self.use_cache and self.cache_dir.exists():
            cache_index_path = self.cache_dir / f"{self.split}_index.pt"
            if cache_index_path.exists():
                cached_index = torch.load(cache_index_path)
                # Validate cache entries exist
                valid_entries = []
                for entry in cached_index:
                    cache_path = Path(entry["cache_path"])
                    if cache_path.exists():
                        valid_entries.append(entry)
                if valid_entries:
                    logger.info(f"Loaded {len(valid_entries)} frames from cache index")
                    if max_segments:
                        # Filter to max segments
                        segments = set()
                        filtered = []
                        for entry in valid_entries:
                            if len(segments) >= max_segments:
                                if entry["segment_id"] in segments:
                                    filtered.append(entry)
                            else:
                                segments.add(entry["segment_id"])
                                filtered.append(entry)
                        return filtered
                    return valid_entries
        
        # Build index from raw data
        logger.info(f"Building frame index from raw data...")
        data_format = self._detect_format()
        
        if data_format == "parquet":
            frame_index = self._build_index_parquet(max_segments)
        elif data_format == "tfrecord":
            frame_index = self._build_index_tfrecord(max_segments)
        elif data_format == "cache":
            # Cache-only mode
            frame_index = self._build_index_cache(max_segments)
        
        return frame_index
    
    def _build_index_parquet(
        self, 
        max_segments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Build index from Parquet format (v2.0.1)."""
        import pyarrow.parquet as pq
        
        frame_index = []
        camera_dir = self.data_dir / self.split / "camera_image"
        
        # Get unique segment IDs
        segment_files = sorted(camera_dir.glob("*.parquet"))
        segments = set()
        
        for pq_file in segment_files:
            # Parse segment ID from filename
            # Format: camera_image_{segment_id}_{shard}.parquet
            parts = pq_file.stem.split("_")
            if len(parts) >= 3:
                segment_id = "_".join(parts[2:-1])
                segments.add(segment_id)
        
        segments = sorted(list(segments))
        if max_segments:
            segments = segments[:max_segments]
        
        logger.info(f"Found {len(segments)} segments")
        
        # Build frame index for each segment
        for segment_id in segments:
            # Get frame count from camera_image table
            segment_frames = []
            for pq_file in camera_dir.glob(f"*{segment_id}*.parquet"):
                table = pq.read_table(pq_file, columns=["key.frame_timestamp_micros"])
                timestamps = table["key.frame_timestamp_micros"].to_pylist()
                segment_frames.extend(timestamps)
            
            unique_timestamps = sorted(set(segment_frames))
            
            for idx, timestamp in enumerate(unique_timestamps):
                cache_path = self.cache_dir / f"{segment_id}_{timestamp}.pt"
                frame_index.append({
                    "segment_id": segment_id,
                    "frame_idx": idx,
                    "timestamp": timestamp,
                    "cache_path": str(cache_path),
                    "cached": cache_path.exists(),
                })
        
        return frame_index
    
    def _build_index_tfrecord(
        self, 
        max_segments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Build index from TFRecord format (v1.4.3)."""
        frame_index = []
        tfrecord_dir = self.data_dir / self.split
        
        tfrecord_files = sorted(tfrecord_dir.glob("*.tfrecord*"))
        if max_segments:
            tfrecord_files = tfrecord_files[:max_segments]
        
        logger.info(f"Found {len(tfrecord_files)} TFRecord files")
        
        for tfrecord_path in tfrecord_files:
            segment_id = tfrecord_path.stem.replace(".tfrecord", "")
            
            # Count frames in TFRecord (requires TensorFlow)
            try:
                import tensorflow as tf
                dataset = tf.data.TFRecordDataset(str(tfrecord_path))
                num_frames = sum(1 for _ in dataset)
            except Exception as e:
                logger.warning(f"Could not count frames in {tfrecord_path}: {e}")
                num_frames = 200  # Assume ~200 frames per segment
            
            for idx in range(num_frames):
                cache_path = self.cache_dir / f"{segment_id}_{idx:04d}.pt"
                frame_index.append({
                    "segment_id": segment_id,
                    "frame_idx": idx,
                    "tfrecord_path": str(tfrecord_path),
                    "cache_path": str(cache_path),
                    "cached": cache_path.exists(),
                })
        
        return frame_index
    
    def _build_index_cache(
        self, 
        max_segments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Build index from cached files only."""
        frame_index = []
        cache_files = sorted(self.cache_dir.glob("*.pt"))
        
        segments = {}
        for cache_file in cache_files:
            # Parse segment_id from filename
            parts = cache_file.stem.rsplit("_", 1)
            if len(parts) == 2:
                segment_id = parts[0]
                if segment_id not in segments:
                    segments[segment_id] = []
                segments[segment_id].append(cache_file)
        
        segment_ids = sorted(segments.keys())
        if max_segments:
            segment_ids = segment_ids[:max_segments]
        
        for segment_id in segment_ids:
            for idx, cache_file in enumerate(sorted(segments[segment_id])):
                frame_index.append({
                    "segment_id": segment_id,
                    "frame_idx": idx,
                    "cache_path": str(cache_file),
                    "cached": True,
                })
        
        return frame_index
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single frame of data.
        
        Returns dict with:
        - images: [N, 3, H, W] camera images
        - intrinsics: [N, 3, 3] camera intrinsic matrices
        - extrinsics: [N, 4, 4] camera-to-ego transformation matrices
        - points: [P, 4] LiDAR points (x, y, z, intensity)
        - ego_pose: [4, 4] ego vehicle pose in world frame
        - boxes_3d: [M, 9] 3D boxes (x, y, z, l, w, h, yaw, class, track_id)
        - waypoints: [K, 3] future ego waypoints (x, y, heading)
        - meta: dict with frame metadata
        """
        frame_info = self.frame_index[idx]
        
        # Try to load from cache
        if frame_info.get("cached", False) and self.use_cache:
            try:
                data = torch.load(frame_info["cache_path"])
                if self.transform:
                    data = self._apply_transforms(data)
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {frame_info['cache_path']}: {e}")
        
        # Load from raw data
        data = self._load_raw_frame(frame_info)
        
        if self.return_raw:
            return data
        
        if self.transform:
            data = self._apply_transforms(data)
        
        return data
    
    def _load_raw_frame(self, frame_info: Dict) -> Dict[str, Any]:
        """Load raw frame data from Waymo dataset."""
        if "tfrecord_path" in frame_info:
            return self._load_from_tfrecord(frame_info)
        else:
            return self._load_from_parquet(frame_info)
    
    def _load_from_parquet(self, frame_info: Dict) -> Dict[str, Any]:
        """Load frame from Parquet format."""
        segment_id = frame_info["segment_id"]
        timestamp = frame_info["timestamp"]
        
        # Load camera images
        images = []
        intrinsics = []
        extrinsics = []
        
        for camera_name in self.config.cameras:
            cam_data = self.parser.load_camera_image(
                self.data_dir / self.split,
                segment_id,
                timestamp,
                self.CAMERA_NAMES[camera_name],
            )
            images.append(cam_data["image"])
            intrinsics.append(cam_data["intrinsic"])
            extrinsics.append(cam_data["extrinsic"])
        
        # Load LiDAR points
        points = self.parser.load_lidar_points(
            self.data_dir / self.split,
            segment_id,
            timestamp,
        )
        
        # Load ego pose
        ego_pose = self.parser.load_ego_pose(
            self.data_dir / self.split,
            segment_id,
            timestamp,
        )
        
        # Load 3D labels
        boxes_3d = self.parser.load_3d_labels(
            self.data_dir / self.split,
            segment_id,
            timestamp,
        )
        
        # Load future ego poses for waypoint labels
        future_poses = self.parser.load_future_poses(
            self.data_dir / self.split,
            segment_id,
            timestamp,
            num_future=self.config.num_future_frames,
        )
        
        # Compute waypoints from future poses
        waypoints = get_future_waypoints_from_poses(
            ego_pose,
            future_poses,
            horizons=np.arange(1, self.config.num_future_frames + 1) * self.config.frame_interval,
        )
        
        return {
            "images": np.stack(images, axis=0),
            "intrinsics": np.stack(intrinsics, axis=0),
            "extrinsics": np.stack(extrinsics, axis=0),
            "points": points,
            "ego_pose": ego_pose,
            "boxes_3d": boxes_3d,
            "waypoints": waypoints,
            "meta": {
                "segment_id": segment_id,
                "timestamp": timestamp,
                "frame_idx": frame_info["frame_idx"],
                "cameras": self.config.cameras,
            },
        }
    
    def _load_from_tfrecord(self, frame_info: Dict) -> Dict[str, Any]:
        """Load frame from TFRecord format."""
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.utils import frame_utils
        
        tfrecord_path = frame_info["tfrecord_path"]
        frame_idx = frame_info["frame_idx"]
        
        # Load the specific frame
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for i, data in enumerate(dataset):
            if i == frame_idx:
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                break
        
        return self._parse_waymo_frame(frame, frame_info)
    
    def _parse_waymo_frame(self, frame, frame_info: Dict) -> Dict[str, Any]:
        """Parse a Waymo Frame protobuf into our format."""
        from waymo_open_dataset.utils import frame_utils
        
        # Extract camera images
        images = []
        intrinsics = []
        extrinsics = []
        
        for camera_name in self.config.cameras:
            cam_idx = self.CAMERA_NAMES[camera_name] - 1
            
            # Get camera image
            camera_image = frame.images[cam_idx]
            image = np.array(Image.open(
                io.BytesIO(camera_image.image)
            ))
            images.append(image)
            
            # Get camera calibration
            camera_calibration = frame.context.camera_calibrations[cam_idx]
            
            # Build intrinsic matrix
            fx = camera_calibration.intrinsic[0]
            fy = camera_calibration.intrinsic[1]
            cx = camera_calibration.intrinsic[2]
            cy = camera_calibration.intrinsic[3]
            intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float32)
            intrinsics.append(intrinsic)
            
            # Build extrinsic matrix (camera to vehicle)
            extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)
            extrinsics.append(extrinsic.astype(np.float32))
        
        # Extract LiDAR points
        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        
        points_all, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
        )
        
        # Merge all LiDAR returns
        points = np.concatenate(points_all, axis=0).astype(np.float32)
        
        # Add intensity (use ones if not available)
        if points.shape[1] == 3:
            intensity = np.ones((points.shape[0], 1), dtype=np.float32)
            points = np.concatenate([points, intensity], axis=1)
        
        # Extract ego pose
        ego_pose = np.array(frame.pose.transform).reshape(4, 4).astype(np.float32)
        
        # Extract 3D labels
        boxes_3d = []
        for label in frame.laser_labels:
            if label.type in self.CLASS_MAP:
                box = np.array([
                    label.box.center_x,
                    label.box.center_y,
                    label.box.center_z,
                    label.box.length,
                    label.box.width,
                    label.box.height,
                    label.box.heading,
                    self.CLASS_MAP[label.type],
                    label.id.encode() if hasattr(label.id, 'encode') else hash(str(label.id)) % 10000,
                ], dtype=np.float32)
                boxes_3d.append(box)
        
        boxes_3d = np.array(boxes_3d, dtype=np.float32) if boxes_3d else np.zeros((0, 9), dtype=np.float32)
        
        # Waypoints will be computed in the cache builder using future frames
        # For now, return empty waypoints
        waypoints = np.zeros((self.config.num_future_frames, 3), dtype=np.float32)
        
        return {
            "images": np.stack(images, axis=0),
            "intrinsics": np.stack(intrinsics, axis=0),
            "extrinsics": np.stack(extrinsics, axis=0),
            "points": points,
            "ego_pose": ego_pose,
            "boxes_3d": boxes_3d,
            "waypoints": waypoints,
            "meta": {
                "segment_id": frame_info["segment_id"],
                "timestamp": frame.timestamp_micros,
                "frame_idx": frame_info["frame_idx"],
                "cameras": self.config.cameras,
            },
        }
    
    def _apply_transforms(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Apply data transforms for model input."""
        # Transform images
        images = []
        for i in range(data["images"].shape[0]):
            img = data["images"][i]
            img_tensor, scale = self.image_transform(img)
            images.append(img_tensor)
        
        # Adjust intrinsics for image scaling
        intrinsics = data["intrinsics"].copy()
        orig_h, orig_w = data["images"].shape[1:3]
        new_w, new_h = self.config.image_size
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        intrinsics[:, 0, 0] *= scale_x  # fx
        intrinsics[:, 1, 1] *= scale_y  # fy
        intrinsics[:, 0, 2] *= scale_x  # cx
        intrinsics[:, 1, 2] *= scale_y  # cy
        
        # Transform point cloud
        points = self.point_cloud_transform(data["points"])
        
        return {
            "images": torch.stack(images, dim=0),
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(data["extrinsics"]),
            "points": points,
            "ego_pose": torch.from_numpy(data["ego_pose"]),
            "boxes_3d": torch.from_numpy(data["boxes_3d"]),
            "waypoints": torch.from_numpy(data["waypoints"]),
            "meta": data["meta"],
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        # Stack tensors
        images = torch.stack([b["images"] for b in batch])
        intrinsics = torch.stack([b["intrinsics"] for b in batch])
        extrinsics = torch.stack([b["extrinsics"] for b in batch])
        ego_poses = torch.stack([b["ego_pose"] for b in batch])
        waypoints = torch.stack([b["waypoints"] for b in batch])
        
        # Pad points and boxes
        max_points = max(b["points"].shape[0] for b in batch)
        max_boxes = max(b["boxes_3d"].shape[0] for b in batch)
        
        points_padded = []
        points_mask = []
        for b in batch:
            n_points = b["points"].shape[0]
            pad_points = torch.zeros(max_points - n_points, 4)
            points_padded.append(torch.cat([b["points"], pad_points], dim=0))
            mask = torch.zeros(max_points, dtype=torch.bool)
            mask[:n_points] = True
            points_mask.append(mask)
        
        boxes_padded = []
        boxes_mask = []
        for b in batch:
            n_boxes = b["boxes_3d"].shape[0]
            pad_boxes = torch.zeros(max_boxes - n_boxes, 9)
            boxes_padded.append(torch.cat([b["boxes_3d"], pad_boxes], dim=0))
            mask = torch.zeros(max_boxes, dtype=torch.bool)
            mask[:n_boxes] = True
            boxes_mask.append(mask)
        
        meta = [b["meta"] for b in batch]
        
        return {
            "images": images,                           # [B, N, 3, H, W]
            "intrinsics": intrinsics,                   # [B, N, 3, 3]
            "extrinsics": extrinsics,                   # [B, N, 4, 4]
            "points": torch.stack(points_padded),       # [B, P, 4]
            "points_mask": torch.stack(points_mask),    # [B, P]
            "ego_pose": ego_poses,                      # [B, 4, 4]
            "boxes_3d": torch.stack(boxes_padded),      # [B, M, 9]
            "boxes_mask": torch.stack(boxes_mask),      # [B, M]
            "waypoints": waypoints,                     # [B, K, 3]
            "meta": meta,
        }
