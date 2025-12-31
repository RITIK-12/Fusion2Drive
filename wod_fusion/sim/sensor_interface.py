"""
Sensor interface for CARLA simulation.

Handles sensor configuration, data preprocessing, and
coordinate transformations to match Waymo format.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    
    name: str = "FRONT"
    width: int = 1920
    height: int = 1280
    fov: float = 70.0
    
    # Transform relative to ego vehicle
    location: Tuple[float, float, float] = (2.0, 0.0, 1.5)  # x, y, z
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    
    # Post-processing
    gamma: float = 2.2
    motion_blur: float = 0.0
    lens_distortion: float = 0.0


@dataclass
class LiDARConfig:
    """Configuration for LiDAR sensor."""
    
    channels: int = 64
    range: float = 75.0
    points_per_second: int = 2200000
    rotation_frequency: float = 20.0
    upper_fov: float = 3.0
    lower_fov: float = -25.0
    
    # Transform relative to ego vehicle
    location: Tuple[float, float, float] = (0.0, 0.0, 2.4)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Noise model
    noise_stddev: float = 0.02  # meters
    dropout_rate: float = 0.0


@dataclass
class SensorConfig:
    """Complete sensor suite configuration."""
    
    # Camera setup (matching Waymo 5-camera layout)
    cameras: List[CameraConfig] = field(default_factory=lambda: [
        CameraConfig(name="FRONT", location=(2.0, 0.0, 1.5), rotation=(0, 0, 0)),
        CameraConfig(name="FRONT_LEFT", location=(1.5, -0.8, 1.5), rotation=(0, -45, 0)),
        CameraConfig(name="FRONT_RIGHT", location=(1.5, 0.8, 1.5), rotation=(0, 45, 0)),
        CameraConfig(name="SIDE_LEFT", location=(0.0, -1.0, 1.5), rotation=(0, -90, 0)),
        CameraConfig(name="SIDE_RIGHT", location=(0.0, 1.0, 1.5), rotation=(0, 90, 0)),
    ])
    
    lidar: LiDARConfig = field(default_factory=LiDARConfig)
    
    # Output settings
    target_image_size: Tuple[int, int] = (256, 704)  # H, W for model input
    max_points: int = 150000
    point_cloud_range: Tuple[float, ...] = (-75.0, -75.0, -2.0, 75.0, 75.0, 4.0)


class SensorInterface:
    """
    Interface for managing and preprocessing sensor data.
    
    Handles:
    - Sensor data collection from CARLA
    - Image preprocessing (resize, normalize)
    - Point cloud preprocessing (range filter, downsample)
    - Calibration matrix computation
    - Coordinate transformations
    """
    
    def __init__(self, config: Optional[SensorConfig] = None):
        """
        Initialize sensor interface.
        
        Args:
            config: Sensor configuration
        """
        self.config = config or SensorConfig()
        
        # Precompute intrinsics
        self._intrinsics = self._compute_intrinsics()
        
        # Precompute extrinsics
        self._extrinsics = self._compute_extrinsics()
    
    def _compute_intrinsics(self) -> Dict[str, np.ndarray]:
        """Compute camera intrinsic matrices."""
        intrinsics = {}
        
        for cam in self.config.cameras:
            # Focal length from FOV
            fov_rad = np.radians(cam.fov)
            fx = cam.width / (2 * np.tan(fov_rad / 2))
            fy = fx  # Square pixels
            
            # Principal point at center
            cx = cam.width / 2
            cy = cam.height / 2
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            intrinsics[cam.name] = K
        
        return intrinsics
    
    def _compute_extrinsics(self) -> Dict[str, np.ndarray]:
        """Compute camera extrinsic matrices (camera to ego)."""
        extrinsics = {}
        
        for cam in self.config.cameras:
            # Translation
            t = np.array(cam.location, dtype=np.float32)
            
            # Rotation (pitch, yaw, roll in degrees)
            pitch, yaw, roll = np.radians(cam.rotation)
            
            # Rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])
            
            Ry = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])
            
            Rz = np.array([
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1]
            ])
            
            R = Rz @ Ry @ Rx
            
            # 4x4 extrinsic matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t
            
            extrinsics[cam.name] = T
        
        return extrinsics
    
    def get_intrinsics(self) -> np.ndarray:
        """
        Get stacked intrinsic matrices for all cameras.
        
        Returns:
            (N, 3, 3) array of intrinsic matrices
        """
        matrices = [self._intrinsics[cam.name] for cam in self.config.cameras]
        return np.stack(matrices, axis=0)
    
    def get_extrinsics(self) -> np.ndarray:
        """
        Get stacked extrinsic matrices for all cameras.
        
        Returns:
            (N, 4, 4) array of extrinsic matrices
        """
        matrices = [self._extrinsics[cam.name] for cam in self.config.cameras]
        return np.stack(matrices, axis=0)
    
    def process_images(
        self,
        images: Dict[str, np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Process camera images for model input.
        
        Args:
            images: Dict mapping camera name to HWC image array
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            (N, 3, H, W) array of processed images
        """
        processed = []
        target_h, target_w = self.config.target_image_size
        
        for cam in self.config.cameras:
            img = images.get(f"camera_{cam.name}")
            
            if img is None:
                # Create dummy image
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            else:
                # Resize
                img = self._resize_image(img, target_h, target_w)
            
            # Normalize
            if normalize:
                img = img.astype(np.float32) / 255.0
            
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            processed.append(img)
        
        return np.stack(processed, axis=0)
    
    def _resize_image(
        self,
        image: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Resize image using simple interpolation."""
        try:
            import cv2
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback to numpy (nearest neighbor)
            h, w = image.shape[:2]
            y_indices = np.linspace(0, h - 1, target_h).astype(int)
            x_indices = np.linspace(0, w - 1, target_w).astype(int)
            return image[np.ix_(y_indices, x_indices)]
    
    def process_lidar(
        self,
        points: np.ndarray,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process LiDAR point cloud for model input.
        
        Args:
            points: (P, 4) array of x, y, z, intensity
            add_noise: Whether to add sensor noise
            
        Returns:
            points: (max_points, 4) padded array
            mask: (max_points,) valid point mask
        """
        # Range filter
        pc_range = self.config.point_cloud_range
        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        )
        points = points[mask]
        
        # Add noise
        if add_noise and self.config.lidar.noise_stddev > 0:
            noise = np.random.randn(*points[:, :3].shape) * self.config.lidar.noise_stddev
            points[:, :3] += noise
        
        # Apply dropout
        if add_noise and self.config.lidar.dropout_rate > 0:
            keep_mask = np.random.rand(len(points)) > self.config.lidar.dropout_rate
            points = points[keep_mask]
        
        # Downsample if too many points
        max_points = self.config.max_points
        
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # Pad to fixed size
        num_points = len(points)
        padded_points = np.zeros((max_points, 4), dtype=np.float32)
        padded_points[:num_points] = points
        
        # Create mask
        mask = np.zeros(max_points, dtype=bool)
        mask[:num_points] = True
        
        return padded_points, mask
    
    def prepare_model_input(
        self,
        sensor_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Prepare complete model input from sensor data.
        
        Args:
            sensor_data: Dict with camera and LiDAR data
            
        Returns:
            Dict with images, intrinsics, extrinsics, points, points_mask
        """
        # Process images
        images = self.process_images(sensor_data)
        
        # Process LiDAR
        lidar_data = sensor_data.get("lidar")
        if lidar_data is not None:
            points, points_mask = self.process_lidar(lidar_data)
        else:
            points = np.zeros((self.config.max_points, 4), dtype=np.float32)
            points_mask = np.zeros(self.config.max_points, dtype=bool)
        
        # Get calibration
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()
        
        # Scale intrinsics for resized images
        scale_x = self.config.target_image_size[1] / self.config.cameras[0].width
        scale_y = self.config.target_image_size[0] / self.config.cameras[0].height
        
        scaled_intrinsics = intrinsics.copy()
        scaled_intrinsics[:, 0, :] *= scale_x  # fx, cx
        scaled_intrinsics[:, 1, :] *= scale_y  # fy, cy
        
        return {
            "images": images,  # (N, 3, H, W)
            "intrinsics": scaled_intrinsics,  # (N, 3, 3)
            "extrinsics": extrinsics,  # (N, 4, 4)
            "points": points,  # (P, 4)
            "points_mask": points_mask,  # (P,)
        }
    
    def carla_to_waymo_coords(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from CARLA to Waymo coordinate system.
        
        CARLA: X-forward, Y-right, Z-up (right-handed)
        Waymo: X-forward, Y-left, Z-up (right-handed)
        
        Args:
            points: (N, 3) or (N, 4) points in CARLA coords
            
        Returns:
            Points in Waymo coords
        """
        result = points.copy()
        result[:, 1] = -result[:, 1]  # Flip Y axis
        return result
    
    def waymo_to_carla_coords(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from Waymo to CARLA coordinate system.
        
        Args:
            points: (N, 3) or (N, 4) points in Waymo coords
            
        Returns:
            Points in CARLA coords
        """
        result = points.copy()
        result[:, 1] = -result[:, 1]  # Flip Y axis
        return result
    
    def project_to_image(
        self,
        points_3d: np.ndarray,
        camera_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: (N, 3) points in ego frame
            camera_name: Camera name
            
        Returns:
            points_2d: (N, 2) image coordinates
            valid_mask: (N,) mask for points in front of camera
        """
        K = self._intrinsics[camera_name]
        T = self._extrinsics[camera_name]
        
        # Transform to camera frame (invert extrinsic)
        T_inv = np.linalg.inv(T)
        
        points_homo = np.concatenate([
            points_3d,
            np.ones((len(points_3d), 1))
        ], axis=1)
        
        points_cam = (T_inv @ points_homo.T).T[:, :3]
        
        # Check if in front of camera
        valid_mask = points_cam[:, 2] > 0.1
        
        # Project
        points_proj = (K @ points_cam.T).T
        points_2d = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-6)
        
        # Check if in image bounds
        cam_config = next(c for c in self.config.cameras if c.name == camera_name)
        in_bounds = (
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < cam_config.width) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < cam_config.height)
        )
        
        valid_mask = valid_mask & in_bounds
        
        return points_2d, valid_mask


class SensorSynchronizer:
    """
    Synchronize sensor data across multiple sensors.
    
    Handles timestamp alignment and interpolation for
    sensors with different update rates.
    """
    
    def __init__(self, sensor_names: List[str], buffer_size: int = 100):
        """
        Args:
            sensor_names: List of sensor names to synchronize
            buffer_size: Size of circular buffer per sensor
        """
        self.sensor_names = sensor_names
        self.buffer_size = buffer_size
        
        self.buffers = {name: [] for name in sensor_names}
    
    def add_data(self, sensor_name: str, timestamp: float, data: Any):
        """Add sensor data to buffer."""
        if sensor_name not in self.buffers:
            return
        
        self.buffers[sensor_name].append((timestamp, data))
        
        # Trim buffer
        if len(self.buffers[sensor_name]) > self.buffer_size:
            self.buffers[sensor_name].pop(0)
    
    def get_synchronized(
        self,
        target_timestamp: float,
        max_time_diff: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """
        Get synchronized sensor data for target timestamp.
        
        Args:
            target_timestamp: Target time to synchronize to
            max_time_diff: Maximum allowed time difference
            
        Returns:
            Dict of sensor data or None if sync fails
        """
        synchronized = {}
        
        for name in self.sensor_names:
            buffer = self.buffers[name]
            
            if not buffer:
                return None
            
            # Find closest timestamp
            closest_idx = min(
                range(len(buffer)),
                key=lambda i: abs(buffer[i][0] - target_timestamp)
            )
            
            timestamp, data = buffer[closest_idx]
            
            if abs(timestamp - target_timestamp) > max_time_diff:
                logger.warning(
                    f"Large time difference for {name}: "
                    f"{abs(timestamp - target_timestamp):.3f}s"
                )
            
            synchronized[name] = data
        
        return synchronized
    
    def clear(self):
        """Clear all buffers."""
        for name in self.sensor_names:
            self.buffers[name] = []
