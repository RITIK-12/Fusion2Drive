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
        """
        Load LiDAR point cloud from Waymo v2 Parquet format.
        
        Waymo v2 stores LiDAR data with files named: {segment_id}.parquet
        Columns include:
        - key.frame_timestamp_micros: timestamp
        - key.laser_name: which LiDAR sensor (1=TOP, 2-5=side LiDARs)
        - [LiDARComponent].range_image_return1.values: range image data
        """
        import pyarrow.parquet as pq
        
        lidar_dir = data_dir / "lidar"
        if not lidar_dir.exists():
            raise ValueError(
                f"LiDAR directory not found: {lidar_dir}. "
                "Ensure you downloaded the 'lidar' component."
            )
        
        all_points = []
        
        # Find the parquet file for this segment
        # File is named: {segment_id}.parquet
        segment_file = lidar_dir / f"{segment_id}.parquet"
        
        if not segment_file.exists():
            # Try to find by pattern match
            matching_files = list(lidar_dir.glob(f"*{segment_id}*.parquet"))
            if not matching_files:
                raise ValueError(
                    f"No LiDAR file found for segment {segment_id} in {lidar_dir}"
                )
            segment_file = matching_files[0]
        
        try:
            # Read the parquet file
            table = pq.read_table(segment_file)
            columns = [f.name for f in table.schema]
            
            # Get timestamp column
            if "key.frame_timestamp_micros" not in columns:
                raise ValueError(f"No timestamp column in {segment_file.name}")
            
            timestamps = table["key.frame_timestamp_micros"].to_pandas()
            mask = timestamps == timestamp
            
            if not mask.any():
                raise ValueError(
                    f"Timestamp {timestamp} not found in {segment_file.name}. "
                    f"Available: {sorted(timestamps.unique())[:5]}..."
                )
            
            # Find range image columns
            range_col = None
            for col in columns:
                if "range_image_return1.values" in col:
                    range_col = col
                    break
            
            if range_col is None:
                # Try alternative: look for any values column
                for col in columns:
                    if ".values" in col.lower():
                        range_col = col
                        break
            
            if range_col is None:
                logger.warning(f"No range image column found. Columns: {columns[:10]}")
                raise ValueError(f"No range image data in {segment_file.name}")
            
            # Process matching rows (one per LiDAR sensor per frame)
            for idx in mask[mask].index:
                try:
                    range_data = table[range_col][idx].as_py()
                    
                    if range_data is None or len(range_data) == 0:
                        continue
                    
                    # Convert range image to points
                    points = self._range_image_to_points(range_data)
                    if points is not None and len(points) > 0:
                        all_points.append(points)
                        
                except Exception as e:
                    logger.debug(f"Could not parse LiDAR row {idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Could not read {segment_file}: {e}")
            raise
        
        if all_points:
            combined = np.concatenate(all_points, axis=0)
            logger.debug(f"Loaded {len(combined)} LiDAR points for {segment_id}")
            return combined
        else:
            raise ValueError(
                f"No LiDAR points found for segment {segment_id}, timestamp {timestamp}. "
                f"File: {segment_file.name}"
            )
    
    def _range_image_to_points(
        self,
        range_data: Union[list, bytes],
    ) -> Optional[np.ndarray]:
        """
        Convert Waymo range image data to 3D point cloud.
        
        The range image encodes: range, intensity, elongation, (x, y, z in vehicle frame)
        For Waymo v2, the data may be pre-computed Cartesian coordinates.
        """
        try:
            if isinstance(range_data, bytes):
                # Binary data - decode as float32
                arr = np.frombuffer(range_data, dtype=np.float32)
            elif isinstance(range_data, (list, tuple)):
                arr = np.array(range_data, dtype=np.float32)
            else:
                return None
            
            if len(arr) == 0:
                return None
            
            # Common case: data is [N, 4] or [N, 6] with (x, y, z, intensity, ...)
            if arr.ndim == 1:
                # Try common formats
                if len(arr) % 4 == 0:
                    arr = arr.reshape(-1, 4)
                elif len(arr) % 6 == 0:
                    arr = arr.reshape(-1, 6)
                elif len(arr) % 3 == 0:
                    arr = arr.reshape(-1, 3)
                else:
                    return None
            
            if arr.ndim != 2:
                # Flatten to 2D if needed
                if arr.ndim > 2:
                    arr = arr.reshape(-1, arr.shape[-1])
            
            if arr.shape[1] < 3:
                return None
            
            # Extract x, y, z (and intensity if available)
            if arr.shape[1] >= 4:
                # Filter out invalid points (range = 0 or -1)
                valid_mask = (arr[:, 0] != 0) & (arr[:, 0] != -1)
                points = arr[valid_mask, :4]  # x, y, z, intensity
            else:
                valid_mask = (arr[:, 0] != 0) & (arr[:, 0] != -1)
                xyz = arr[valid_mask, :3]
                intensity = np.ones((len(xyz), 1), dtype=np.float32)
                points = np.concatenate([xyz, intensity], axis=1)
            
            return points if len(points) > 0 else None
            
        except Exception as e:
            logger.debug(f"Range image conversion failed: {e}")
            return None
    
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
