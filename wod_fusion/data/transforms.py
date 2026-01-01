"""
Data transforms for image and point cloud processing.
"""

from typing import Tuple, List, Optional, Union
import numpy as np
import torch
import cv2


class ImageTransform:
    """
    Image preprocessing transforms.
    
    Applies:
    - Resize to target size
    - Normalization (ImageNet stats)
    - Optional augmentation for training
    """
    
    # ImageNet normalization stats
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 480),
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        Initialize transform.
        
        Args:
            target_size: (width, height) for resizing
            normalize: Whether to apply ImageNet normalization
            augment: Whether to apply data augmentation (training only)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
    
    def __call__(
        self, 
        image: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """
        Transform image.
        
        Args:
            image: Input image [H, W, 3] in uint8
            
        Returns:
            Tuple of:
            - Transformed image tensor [3, H, W]
            - Scale factors (scale_x, scale_y)
        """
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Resize
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Convert to float
        image = image.astype(np.float32) / 255.0
        
        # Augmentation
        if self.augment:
            image = self._apply_augmentation(image)
        
        # Normalize
        if self.normalize:
            image = (image - self.MEAN) / self.STD
        
        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image.transpose(2, 0, 1))
        
        return image, (scale_x, scale_y)
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Color jitter
        if np.random.rand() < 0.5:
            # Brightness
            delta = np.random.uniform(-0.2, 0.2)
            image = np.clip(image + delta, 0, 1)
        
        if np.random.rand() < 0.5:
            # Contrast
            alpha = np.random.uniform(0.8, 1.2)
            image = np.clip(image * alpha, 0, 1)
        
        return image
    
    @staticmethod
    def denormalize(image: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor back to uint8 image for visualization."""
        if image.dim() == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        
        image = image * ImageTransform.STD + ImageTransform.MEAN
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


class PointCloudTransform:
    """
    Point cloud preprocessing transforms.
    
    Applies:
    - Range filtering
    - Random sampling / padding to fixed size
    - Optional augmentation for training
    """
    
    def __init__(
        self,
        point_cloud_range: List[float] = None,
        max_points: int = 150000,
        augment: bool = False,
    ):
        """
        Initialize transform.
        
        Args:
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points: Maximum number of points to return
            augment: Whether to apply data augmentation
        """
        if point_cloud_range is None:
            point_cloud_range = [-75.0, -75.0, -3.0, 75.0, 75.0, 5.0]
        
        self.range = np.array(point_cloud_range, dtype=np.float32)
        self.max_points = max_points
        self.augment = augment
    
    def __call__(self, points: np.ndarray) -> torch.Tensor:
        """
        Transform point cloud.
        
        Args:
            points: Input points [N, 4+] (x, y, z, intensity, ...)
            
        Returns:
            Transformed points tensor [M, 4]
        """
        # Ensure we have at least 4 columns
        if points.shape[1] < 4:
            padding = np.zeros((points.shape[0], 4 - points.shape[1]), dtype=np.float32)
            points = np.concatenate([points, padding], axis=1)
        
        # Keep only x, y, z, intensity
        points = points[:, :4].astype(np.float32)
        
        # Range filter
        mask = (
            (points[:, 0] >= self.range[0]) & (points[:, 0] <= self.range[3]) &
            (points[:, 1] >= self.range[1]) & (points[:, 1] <= self.range[4]) &
            (points[:, 2] >= self.range[2]) & (points[:, 2] <= self.range[5])
        )
        points = points[mask]
        
        # Augmentation
        if self.augment:
            points = self._apply_augmentation(points)
        
        # Random sample or pad
        n_points = points.shape[0]
        if n_points > self.max_points:
            indices = np.random.choice(n_points, self.max_points, replace=False)
            points = points[indices]
        elif n_points < self.max_points:
            padding = np.zeros((self.max_points - n_points, 4), dtype=np.float32)
            points = np.concatenate([points, padding], axis=0)
        
        return torch.from_numpy(points)
    
    def _apply_augmentation(self, points: np.ndarray) -> np.ndarray:
        """Apply point cloud augmentation."""
        # Random rotation around z-axis
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-np.pi / 4, np.pi / 4)
            cos, sin = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1],
            ], dtype=np.float32)
            points[:, :3] = points[:, :3] @ rotation.T
        
        # Random flip along x-axis
        if np.random.rand() < 0.5:
            points[:, 1] = -points[:, 1]
        
        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale
        
        # Random translation
        if np.random.rand() < 0.5:
            translation = np.random.uniform(-0.2, 0.2, size=3).astype(np.float32)
            points[:, :3] += translation
        
        return points


class SensorCalibration:
    """
    Handle sensor calibration and coordinate transforms.
    
    Coordinate Frames:
    - Camera frame: x-right, y-down, z-forward
    - LiDAR frame: x-forward, y-left, z-up  
    - Ego frame (vehicle): x-forward, y-left, z-up
    - World frame: arbitrary, but consistent within segment
    """
    
    def __init__(
        self,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image_size: Tuple[int, int],
    ):
        """
        Initialize calibration.
        
        Args:
            intrinsic: 3x3 camera intrinsic matrix
            extrinsic: 4x4 camera-to-ego transformation
            image_size: (width, height) of image
        """
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.image_size = image_size
        
        # Compute ego-to-camera transform
        self.extrinsic_inv = np.linalg.inv(extrinsic)
    
    def project_to_image(
        self, 
        points_ego: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points from ego frame to image plane.
        
        Args:
            points_ego: [N, 3] points in ego frame
            
        Returns:
            Tuple of:
            - [M, 2] image coordinates (u, v)
            - [M] depth values
            - [N] boolean mask of valid projections
        """
        # Transform to camera frame
        points_homo = np.concatenate([
            points_ego,
            np.ones((points_ego.shape[0], 1), dtype=points_ego.dtype)
        ], axis=1)
        points_cam = (self.extrinsic_inv @ points_homo.T).T[:, :3]
        
        # Filter points behind camera
        valid = points_cam[:, 2] > 0.1  # Minimum depth
        
        # Project to image
        points_img = (self.intrinsic @ points_cam.T).T
        depths = points_img[:, 2]
        uv = points_img[:, :2] / depths[:, None]
        
        # Filter points outside image
        w, h = self.image_size
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w)
        valid &= (uv[:, 1] >= 0) & (uv[:, 1] < h)
        
        return uv[valid], depths[valid], valid
    
    def unproject_to_3d(
        self,
        uv: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        """
        Unproject 2D points to 3D in ego frame.
        
        Args:
            uv: [N, 2] image coordinates
            depth: [N] depth values
            
        Returns:
            [N, 3] points in ego frame
        """
        # Unproject to camera frame
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x = (uv[:, 0] - cx) * depth / fx
        y = (uv[:, 1] - cy) * depth / fy
        z = depth
        
        points_cam = np.stack([x, y, z], axis=1)
        
        # Transform to ego frame
        points_homo = np.concatenate([
            points_cam,
            np.ones((points_cam.shape[0], 1), dtype=points_cam.dtype)
        ], axis=1)
        points_ego = (self.extrinsic @ points_homo.T).T[:, :3]
        
        return points_ego
    
    def transform_boxes_to_image(
        self,
        boxes_3d: np.ndarray,
    ) -> np.ndarray:
        """
        Project 3D boxes to 2D bounding boxes in image.
        
        Args:
            boxes_3d: [N, 7] boxes (x, y, z, l, w, h, yaw)
            
        Returns:
            [N, 4] 2D boxes (x1, y1, x2, y2) or NaN for invalid boxes
        """
        from wod_fusion.utils.geometry import get_box_corners
        
        boxes_2d = []
        for box in boxes_3d:
            corners = get_box_corners(box)  # [8, 3]
            uv, _, valid = self.project_to_image(corners)
            
            if valid.sum() >= 4:  # At least 4 corners visible
                x1, y1 = uv.min(axis=0)
                x2, y2 = uv.max(axis=0)
                boxes_2d.append([x1, y1, x2, y2])
            else:
                boxes_2d.append([np.nan, np.nan, np.nan, np.nan])
        
        return np.array(boxes_2d, dtype=np.float32)
