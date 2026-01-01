"""
Geometry utilities for 3D transformations and box operations.
"""

from typing import Tuple, Optional

import numpy as np
import torch


def rotate_points_2d(
    points: torch.Tensor,
    angle: torch.Tensor,
    center: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Rotate 2D points around a center.
    
    Args:
        points: (N, 2) or (B, N, 2) tensor of points
        angle: Rotation angle in radians (scalar or (B,))
        center: Center of rotation (2,) or (B, 2), default is origin
        
    Returns:
        Rotated points with same shape as input
    """
    if center is not None:
        points = points - center
    
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    
    if points.dim() == 2:
        rotation = torch.stack([
            torch.stack([cos, -sin]),
            torch.stack([sin, cos])
        ], dim=0)
        rotated = torch.matmul(points, rotation.T)
    else:
        # Batched rotation
        rotation = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1)
        ], dim=-2)
        rotated = torch.einsum("bni,bij->bnj", points, rotation)
    
    if center is not None:
        rotated = rotated + center
    
    return rotated


def rotate_points_3d(
    points: torch.Tensor,
    angles: torch.Tensor,
    order: str = "zyx",
) -> torch.Tensor:
    """
    Rotate 3D points by Euler angles.
    
    Args:
        points: (N, 3) or (B, N, 3) tensor of points
        angles: (3,) or (B, 3) Euler angles in radians (roll, pitch, yaw)
        order: Rotation order, default "zyx"
        
    Returns:
        Rotated points with same shape as input
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
        angles = angles.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = points.shape[0]
    
    # Build rotation matrices
    roll, pitch, yaw = angles[:, 0], angles[:, 1], angles[:, 2]
    
    # Roll (rotation around X)
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    Rx = torch.zeros(batch_size, 3, 3, device=points.device, dtype=points.dtype)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_r
    Rx[:, 1, 2] = -sin_r
    Rx[:, 2, 1] = sin_r
    Rx[:, 2, 2] = cos_r
    
    # Pitch (rotation around Y)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    Ry = torch.zeros(batch_size, 3, 3, device=points.device, dtype=points.dtype)
    Ry[:, 0, 0] = cos_p
    Ry[:, 0, 2] = sin_p
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_p
    Ry[:, 2, 2] = cos_p
    
    # Yaw (rotation around Z)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    Rz = torch.zeros(batch_size, 3, 3, device=points.device, dtype=points.dtype)
    Rz[:, 0, 0] = cos_y
    Rz[:, 0, 1] = -sin_y
    Rz[:, 1, 0] = sin_y
    Rz[:, 1, 1] = cos_y
    Rz[:, 2, 2] = 1
    
    # Combine rotations
    if order == "zyx":
        R = Rz @ Ry @ Rx
    elif order == "xyz":
        R = Rx @ Ry @ Rz
    else:
        raise ValueError(f"Unknown rotation order: {order}")
    
    # Apply rotation
    rotated = torch.einsum("bni,bij->bnj", points, R)
    
    if squeeze_output:
        rotated = rotated.squeeze(0)
    
    return rotated


def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 4x4 transformation matrix to 3D points.
    
    Args:
        points: (N, 3) or (B, N, 3) points
        transform: (4, 4) or (B, 4, 4) transformation matrix
        
    Returns:
        Transformed points with same shape as input
    """
    if points.dim() == 2:
        # Add homogeneous coordinate
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=-1)
        
        # Transform
        transformed = torch.matmul(points_homo, transform.T)
        return transformed[:, :3]
    else:
        # Batched transformation
        batch_size, num_points, _ = points.shape
        ones = torch.ones(batch_size, num_points, 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=-1)
        
        # Transform: (B, N, 4) x (B, 4, 4)^T -> (B, N, 4)
        transformed = torch.einsum("bni,bij->bnj", points_homo, transform)
        return transformed[:, :, :3]


def corners_from_box_3d(
    boxes: np.ndarray,
) -> np.ndarray:
    """
    Get 8 corner points from 3D boxes.
    
    Args:
        boxes: (N, 7) boxes with [x, y, z, length, width, height, heading]
        
    Returns:
        corners: (N, 8, 3) corner points
    """
    centers = boxes[:, :3]
    dims = boxes[:, 3:6]  # length, width, height
    headings = boxes[:, 6]
    
    # Half dimensions
    l, w, h = dims[:, 0:1] / 2, dims[:, 1:2] / 2, dims[:, 2:3] / 2
    
    # 8 corners in local frame (before rotation)
    # Order: front-left-bottom, front-right-bottom, rear-right-bottom, rear-left-bottom,
    #        front-left-top, front-right-top, rear-right-top, rear-left-top
    corners_local = np.stack([
        np.concatenate([l, w, -h], axis=1),
        np.concatenate([l, -w, -h], axis=1),
        np.concatenate([-l, -w, -h], axis=1),
        np.concatenate([-l, w, -h], axis=1),
        np.concatenate([l, w, h], axis=1),
        np.concatenate([l, -w, h], axis=1),
        np.concatenate([-l, -w, h], axis=1),
        np.concatenate([-l, w, h], axis=1),
    ], axis=1)  # (N, 8, 3)
    
    # Rotation matrices
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    
    # Rotate around Z axis
    corners = np.zeros_like(corners_local)
    corners[:, :, 0] = corners_local[:, :, 0] * cos_h[:, None] - corners_local[:, :, 1] * sin_h[:, None]
    corners[:, :, 1] = corners_local[:, :, 0] * sin_h[:, None] + corners_local[:, :, 1] * cos_h[:, None]
    corners[:, :, 2] = corners_local[:, :, 2]
    
    # Translate
    corners = corners + centers[:, None, :]
    
    return corners


def corners_from_box_3d_torch(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    Get 8 corner points from 3D boxes (PyTorch version).
    
    Args:
        boxes: (N, 7) or (B, N, 7) boxes with [x, y, z, length, width, height, heading]
        
    Returns:
        corners: (N, 8, 3) or (B, N, 8, 3) corner points
    """
    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, num_boxes, _ = boxes.shape
    
    centers = boxes[:, :, :3]
    dims = boxes[:, :, 3:6]
    headings = boxes[:, :, 6]
    
    l = dims[:, :, 0:1] / 2
    w = dims[:, :, 1:2] / 2
    h = dims[:, :, 2:3] / 2
    
    # 8 corners
    corners_local = torch.stack([
        torch.cat([l, w, -h], dim=-1),
        torch.cat([l, -w, -h], dim=-1),
        torch.cat([-l, -w, -h], dim=-1),
        torch.cat([-l, w, -h], dim=-1),
        torch.cat([l, w, h], dim=-1),
        torch.cat([l, -w, h], dim=-1),
        torch.cat([-l, -w, h], dim=-1),
        torch.cat([-l, w, h], dim=-1),
    ], dim=2)  # (B, N, 8, 3)
    
    # Rotation
    cos_h = torch.cos(headings)
    sin_h = torch.sin(headings)
    
    corners = torch.zeros_like(corners_local)
    corners[:, :, :, 0] = corners_local[:, :, :, 0] * cos_h[:, :, None] - corners_local[:, :, :, 1] * sin_h[:, :, None]
    corners[:, :, :, 1] = corners_local[:, :, :, 0] * sin_h[:, :, None] + corners_local[:, :, :, 1] * cos_h[:, :, None]
    corners[:, :, :, 2] = corners_local[:, :, :, 2]
    
    # Translate
    corners = corners + centers[:, :, None, :]
    
    if squeeze_output:
        corners = corners.squeeze(0)
    
    return corners


def box_iou_3d(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    mode: str = "iou",
) -> np.ndarray:
    """
    Compute 3D IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 7) boxes [x, y, z, l, w, h, heading]
        boxes2: (M, 7) boxes [x, y, z, l, w, h, heading]
        mode: "iou" or "giou"
        
    Returns:
        iou: (N, M) IoU matrix
    """
    from shapely.geometry import Polygon
    
    n = len(boxes1)
    m = len(boxes2)
    
    if n == 0 or m == 0:
        return np.zeros((n, m))
    
    # Get corners for BEV IoU
    corners1 = corners_from_box_3d(boxes1)[:, :4, :2]  # Bottom 4 corners, x-y only
    corners2 = corners_from_box_3d(boxes2)[:, :4, :2]
    
    # Heights
    z1 = boxes1[:, 2]
    h1 = boxes1[:, 5]
    z2 = boxes2[:, 2]
    h2 = boxes2[:, 5]
    
    # Compute IoU
    iou = np.zeros((n, m))
    
    for i in range(n):
        poly1 = Polygon(corners1[i])
        area1 = poly1.area
        
        z1_min = z1[i] - h1[i] / 2
        z1_max = z1[i] + h1[i] / 2
        
        for j in range(m):
            poly2 = Polygon(corners2[j])
            area2 = poly2.area
            
            # BEV intersection
            try:
                intersection_2d = poly1.intersection(poly2).area
            except Exception:
                intersection_2d = 0.0
            
            if intersection_2d == 0:
                continue
            
            # Height intersection
            z2_min = z2[j] - h2[j] / 2
            z2_max = z2[j] + h2[j] / 2
            
            z_min = max(z1_min, z2_min)
            z_max = min(z1_max, z2_max)
            
            if z_max <= z_min:
                continue
            
            # 3D intersection
            intersection_3d = intersection_2d * (z_max - z_min)
            
            # 3D volumes
            vol1 = area1 * h1[i]
            vol2 = area2 * h2[j]
            
            # IoU
            union = vol1 + vol2 - intersection_3d
            iou[i, j] = intersection_3d / (union + 1e-8)
    
    return iou


def nms_3d(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    3D Non-Maximum Suppression.
    
    Args:
        boxes: (N, 7) boxes [x, y, z, l, w, h, heading]
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        keep: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    
    # Sort by score
    order = np.argsort(-scores)
    
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        remaining = order[1:]
        ious = box_iou_3d(boxes[i:i+1], boxes[remaining])
        
        # Keep boxes with IoU < threshold
        mask = ious[0] < iou_threshold
        order = remaining[mask]
    
    return np.array(keep, dtype=np.int64)


def project_to_image(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) points in world frame
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) world-to-camera transform
        image_size: (height, width)
        
    Returns:
        points_2d: (N, 2) image coordinates
        valid_mask: (N,) mask for valid projections
    """
    # Transform to camera frame
    ones = np.ones((len(points_3d), 1))
    points_homo = np.concatenate([points_3d, ones], axis=1)
    points_cam = (extrinsics @ points_homo.T).T[:, :3]
    
    # Check if in front of camera
    valid_mask = points_cam[:, 2] > 0.1
    
    # Project
    points_proj = (intrinsics @ points_cam.T).T
    points_2d = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-8)
    
    # Check bounds
    h, w = image_size
    in_bounds = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    )
    
    valid_mask = valid_mask & in_bounds
    
    return points_2d, valid_mask


def get_future_waypoints_from_poses(
    current_pose: np.ndarray,
    future_poses,
    num_waypoints: int = 10,
) -> np.ndarray:
    """
    Extract future waypoints from vehicle poses in local frame.
    
    Args:
        current_pose: (4, 4) current vehicle pose (world frame)
        future_poses: List of (4, 4) or (T, 4, 4) future vehicle poses (world frame)
        num_waypoints: Number of waypoints to extract
        
    Returns:
        waypoints: (num_waypoints, 2) XY waypoints in current vehicle frame
    """
    if len(future_poses) == 0:
        return np.zeros((num_waypoints, 2), dtype=np.float32)
    
    # Convert list to numpy array if needed
    if isinstance(future_poses, list):
        future_poses = np.stack(future_poses, axis=0)
    
    # Get inverse of current pose to transform to local frame
    current_pose_inv = np.linalg.inv(current_pose)
    
    # Extract future positions (translation component)
    future_positions_world = future_poses[:, :3, 3]  # (T, 3)
    
    # Transform to current vehicle frame
    ones = np.ones((len(future_positions_world), 1))
    future_positions_homo = np.concatenate([future_positions_world, ones], axis=1)  # (T, 4)
    future_positions_local = (current_pose_inv @ future_positions_homo.T).T[:, :2]  # (T, 2)
    
    # Interpolate or pad to get desired number of waypoints
    if len(future_positions_local) >= num_waypoints:
        # Sample evenly spaced waypoints
        indices = np.linspace(0, len(future_positions_local) - 1, num_waypoints, dtype=int)
        waypoints = future_positions_local[indices]
    else:
        # Pad with last position
        waypoints = np.zeros((num_waypoints, 2), dtype=np.float32)
        waypoints[:len(future_positions_local)] = future_positions_local
        if len(future_positions_local) > 0:
            waypoints[len(future_positions_local):] = future_positions_local[-1]
    
    return waypoints.astype(np.float32)
