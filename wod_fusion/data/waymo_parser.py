"""
Waymo frame parser for both Parquet and TFRecord formats.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np

logger = logging.getLogger(__name__)


class WaymoFrameParser:
    """
    Parser for Waymo Open Dataset frames.
    
    Supports:
    - v2.0.1 Parquet format (modular, preferred)
    - v1.4.3 TFRecord format (legacy)
    """
    
    def __init__(self, data_format: str = "parquet"):
        """
        Initialize parser.
        
        Args:
            data_format: Either 'parquet' or 'tfrecord'
        """
        self.data_format = data_format
        self._parquet_cache = {}
    
    def load_camera_image(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
        camera_id: int,
    ) -> Dict[str, np.ndarray]:
        """
        Load camera image and calibration.
        
        Args:
            data_dir: Path to split directory
            segment_id: Segment identifier
            timestamp: Frame timestamp in microseconds
            camera_id: Camera ID (1-5)
            
        Returns:
            Dict with 'image', 'intrinsic', 'extrinsic'
        """
        if self.data_format == "parquet":
            return self._load_camera_parquet(data_dir, segment_id, timestamp, camera_id)
        else:
            raise NotImplementedError("Use _parse_waymo_frame for TFRecord format")
    
    def _load_camera_parquet(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
        camera_id: int,
    ) -> Dict[str, np.ndarray]:
        """Load camera from Parquet format."""
        import pyarrow.parquet as pq
        from PIL import Image
        import io
        
        # Find the right parquet file
        camera_dir = data_dir / "camera_image"
        
        # Read camera image
        for pq_file in camera_dir.glob(f"*{segment_id}*.parquet"):
            table = pq.read_table(pq_file)
            
            # Filter by timestamp and camera
            mask = (
                (table["key.frame_timestamp_micros"].to_pandas() == timestamp) &
                (table["key.camera_name"].to_pandas() == camera_id)
            )
            
            if mask.any():
                row_idx = mask.idxmax()
                image_data = table["[CameraImageComponent].image"][row_idx].as_py()
                image = np.array(Image.open(io.BytesIO(image_data)))
                break
        else:
            raise ValueError(f"Camera image not found for {segment_id}, {timestamp}, {camera_id}")
        
        # Load calibration
        calib_dir = data_dir / "camera_calibration"
        intrinsic = np.eye(3, dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        
        for pq_file in calib_dir.glob(f"*{segment_id}*.parquet"):
            table = pq.read_table(pq_file)
            mask = table["key.camera_name"].to_pandas() == camera_id
            
            if mask.any():
                row_idx = mask.idxmax()
                
                # Intrinsic
                fx = table["[CameraCalibrationComponent].intrinsic.f_u"][row_idx].as_py()
                fy = table["[CameraCalibrationComponent].intrinsic.f_v"][row_idx].as_py()
                cx = table["[CameraCalibrationComponent].intrinsic.c_u"][row_idx].as_py()
                cy = table["[CameraCalibrationComponent].intrinsic.c_v"][row_idx].as_py()
                
                intrinsic = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ], dtype=np.float32)
                
                # Extrinsic
                extrinsic_flat = table["[CameraCalibrationComponent].extrinsic.transform"][row_idx].as_py()
                extrinsic = np.array(extrinsic_flat, dtype=np.float32).reshape(4, 4)
                break
        
        return {
            "image": image,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
        }
    
    def load_lidar_points(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
    ) -> np.ndarray:
        """
        Load LiDAR point cloud.
        
        Args:
            data_dir: Path to split directory
            segment_id: Segment identifier
            timestamp: Frame timestamp in microseconds
            
        Returns:
            Points array of shape [N, 4] (x, y, z, intensity)
        """
        if self.data_format == "parquet":
            return self._load_lidar_parquet(data_dir, segment_id, timestamp)
        else:
            raise NotImplementedError("Use _parse_waymo_frame for TFRecord format")
    
    def _load_lidar_parquet(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
    ) -> np.ndarray:
        """Load LiDAR from Parquet format."""
        import pyarrow.parquet as pq
        
        lidar_dir = data_dir / "lidar"
        all_points = []
        
        for pq_file in lidar_dir.glob(f"*{segment_id}*.parquet"):
            table = pq.read_table(pq_file)
            mask = table["key.frame_timestamp_micros"].to_pandas() == timestamp
            
            if mask.any():
                # Get range image data
                for idx in mask[mask].index:
                    try:
                        # Get range image and convert to points
                        range_image = table["[LiDARComponent].range_image_return1.values"][idx].as_py()
                        # Parse range image (simplified - actual parsing is more complex)
                        if range_image:
                            # This is a simplified version - actual implementation needs
                            # proper range image to point cloud conversion
                            points = np.frombuffer(range_image, dtype=np.float32).reshape(-1, 4)
                            all_points.append(points)
                    except Exception as e:
                        logger.debug(f"Could not parse LiDAR data: {e}")
                        continue
        
        if all_points:
            return np.concatenate(all_points, axis=0)
        else:
            # Raise error if no LiDAR points found - do not use synthetic data
            raise ValueError(
                f"No LiDAR points found for segment {segment_id}, timestamp {timestamp}. "
                "Ensure the LiDAR data is properly downloaded and the Parquet files are not corrupted."
            )
    
    def load_ego_pose(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
    ) -> np.ndarray:
        """
        Load ego vehicle pose.
        
        Args:
            data_dir: Path to split directory
            segment_id: Segment identifier
            timestamp: Frame timestamp in microseconds
            
        Returns:
            4x4 transformation matrix (ego to world)
        """
        if self.data_format == "parquet":
            return self._load_ego_pose_parquet(data_dir, segment_id, timestamp)
        else:
            raise NotImplementedError("Use _parse_waymo_frame for TFRecord format")
    
    def _load_ego_pose_parquet(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
    ) -> np.ndarray:
        """Load ego pose from Parquet format."""
        import pyarrow.parquet as pq
        
        pose_dir = data_dir / "vehicle_pose"
        
        for pq_file in pose_dir.glob(f"*{segment_id}*.parquet"):
            table = pq.read_table(pq_file)
            mask = table["key.frame_timestamp_micros"].to_pandas() == timestamp
            
            if mask.any():
                row_idx = mask.idxmax()
                pose_flat = table["[VehiclePoseComponent].world_from_vehicle.transform"][row_idx].as_py()
                return np.array(pose_flat, dtype=np.float32).reshape(4, 4)
        
        # Return identity if not found
        logger.warning(f"Ego pose not found for {segment_id}, {timestamp}")
        return np.eye(4, dtype=np.float32)
    
    def load_3d_labels(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
    ) -> np.ndarray:
        """
        Load 3D object labels.
        
        Args:
            data_dir: Path to split directory
            segment_id: Segment identifier
            timestamp: Frame timestamp in microseconds
            
        Returns:
            Array of shape [N, 9] (x, y, z, l, w, h, yaw, class, track_id)
        """
        if self.data_format == "parquet":
            return self._load_3d_labels_parquet(data_dir, segment_id, timestamp)
        else:
            raise NotImplementedError("Use _parse_waymo_frame for TFRecord format")
    
    def _load_3d_labels_parquet(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
    ) -> np.ndarray:
        """Load 3D labels from Parquet format."""
        import pyarrow.parquet as pq
        
        labels_dir = data_dir / "lidar_box"
        boxes = []
        
        # Class mapping
        CLASS_MAP = {1: 0, 2: 1, 4: 2}  # vehicle, pedestrian, cyclist
        
        for pq_file in labels_dir.glob(f"*{segment_id}*.parquet"):
            table = pq.read_table(pq_file)
            mask = table["key.frame_timestamp_micros"].to_pandas() == timestamp
            
            for idx in mask[mask].index:
                try:
                    obj_type = table["[LiDARBoxComponent].type"][idx].as_py()
                    if obj_type not in CLASS_MAP:
                        continue
                    
                    cx = table["[LiDARBoxComponent].box.center.x"][idx].as_py()
                    cy = table["[LiDARBoxComponent].box.center.y"][idx].as_py()
                    cz = table["[LiDARBoxComponent].box.center.z"][idx].as_py()
                    length = table["[LiDARBoxComponent].box.size.x"][idx].as_py()
                    width = table["[LiDARBoxComponent].box.size.y"][idx].as_py()
                    height = table["[LiDARBoxComponent].box.size.z"][idx].as_py()
                    heading = table["[LiDARBoxComponent].box.heading"][idx].as_py()
                    
                    boxes.append([
                        cx, cy, cz, length, width, height, heading,
                        CLASS_MAP[obj_type], idx,
                    ])
                except Exception as e:
                    logger.debug(f"Could not parse box: {e}")
                    continue
        
        if boxes:
            return np.array(boxes, dtype=np.float32)
        else:
            return np.zeros((0, 9), dtype=np.float32)
    
    def load_future_poses(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
        num_future: int = 10,
    ) -> List[np.ndarray]:
        """
        Load future ego poses for waypoint computation.
        
        Args:
            data_dir: Path to split directory
            segment_id: Segment identifier
            timestamp: Current frame timestamp in microseconds
            num_future: Number of future frames to load
            
        Returns:
            List of 4x4 transformation matrices
        """
        if self.data_format == "parquet":
            return self._load_future_poses_parquet(
                data_dir, segment_id, timestamp, num_future
            )
        else:
            raise NotImplementedError("Use _parse_waymo_frame for TFRecord format")
    
    def _load_future_poses_parquet(
        self,
        data_dir: Path,
        segment_id: str,
        timestamp: int,
        num_future: int,
    ) -> List[np.ndarray]:
        """Load future poses from Parquet format."""
        import pyarrow.parquet as pq
        
        pose_dir = data_dir / "vehicle_pose"
        all_poses = {}
        
        for pq_file in pose_dir.glob(f"*{segment_id}*.parquet"):
            table = pq.read_table(pq_file)
            timestamps = table["key.frame_timestamp_micros"].to_pandas()
            
            for idx, ts in enumerate(timestamps):
                if ts >= timestamp:
                    pose_flat = table["[VehiclePoseComponent].world_from_vehicle.transform"][idx].as_py()
                    all_poses[ts] = np.array(pose_flat, dtype=np.float32).reshape(4, 4)
        
        # Sort by timestamp and take future frames
        sorted_timestamps = sorted([ts for ts in all_poses.keys() if ts > timestamp])
        future_poses = [all_poses[ts] for ts in sorted_timestamps[:num_future]]
        
        # Pad with last known pose if not enough future frames
        while len(future_poses) < num_future:
            if future_poses:
                future_poses.append(future_poses[-1].copy())
            else:
                future_poses.append(np.eye(4, dtype=np.float32))
        
        return future_poses
