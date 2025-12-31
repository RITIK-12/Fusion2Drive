"""
Loss functions for multi-task learning.

Includes:
- Focal loss for heatmap (detection centers)
- L1/Smooth-L1 for box regression
- L1/Smooth-L1 for waypoint regression
- Multi-task loss with configurable weighting
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal loss for dense prediction (heatmap).
    
    Focal Loss = -alpha * (1 - p)^gamma * log(p)   for positive
               = -(1 - alpha) * p^gamma * log(1-p) for negative
    
    Reduces loss for well-classified examples, focusing on hard negatives.
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Focusing parameter for positives (gamma in original paper)
            beta: Focusing parameter for negatives
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: [B, C, H, W] predicted heatmap (after sigmoid)
            target: [B, C, H, W] target heatmap (Gaussian peaks at centers)
            
        Returns:
            Scalar loss
        """
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)
        
        # Positive loss (at object centers)
        pos_mask = target == 1
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        
        # Negative loss (background)
        neg_mask = target < 1
        neg_loss = -((1 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask
        
        # Normalize by number of positives
        num_pos = pos_mask.float().sum()
        
        if num_pos > 0:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            loss = neg_loss.sum()
        
        return loss


class DetectionLoss(nn.Module):
    """
    Combined loss for 3D object detection.
    
    Components:
    1. Heatmap loss (focal loss for center prediction)
    2. Box regression loss (L1 for offset, dimensions, rotation)
    
    Only regression loss is computed at positive locations.
    """
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        dim_weight: float = 1.0,
        rot_weight: float = 1.0,
        height_weight: float = 1.0,
    ):
        """
        Initialize detection loss.
        
        Args:
            heatmap_weight: Weight for heatmap focal loss
            offset_weight: Weight for center offset regression
            dim_weight: Weight for dimension regression
            rot_weight: Weight for rotation regression
            height_weight: Weight for height regression
        """
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=2.0, beta=4.0)
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.dim_weight = dim_weight
        self.rot_weight = rot_weight
        self.height_weight = height_weight
    
    def forward(
        self,
        pred_heatmap: torch.Tensor,
        pred_reg: torch.Tensor,
        target_heatmap: torch.Tensor,
        target_reg: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            pred_heatmap: [B, C, H, W] predicted heatmap (logits)
            pred_reg: [B, 8, H, W] predicted regression
            target_heatmap: [B, C, H, W] target heatmap
            target_reg: [B, 8, H, W] target regression
            target_mask: [B, H, W] mask for valid regression locations
            
        Returns:
            Dict with loss components
        """
        # Apply sigmoid to heatmap
        pred_heatmap_sigmoid = torch.sigmoid(pred_heatmap)
        
        # Heatmap loss
        heatmap_loss = self.focal_loss(pred_heatmap_sigmoid, target_heatmap)
        
        # Regression losses (only at positive locations)
        if target_mask.sum() > 0:
            # Expand mask for 8 channels
            mask = target_mask.unsqueeze(1).expand_as(pred_reg)
            
            # Get predictions and targets at positive locations
            pred_flat = pred_reg[mask]
            target_flat = target_reg[mask]
            
            # Split into components
            # Channels: offset(2), height(1), dim(3), rot(2)
            n_points = pred_flat.numel() // 8
            pred_split = pred_flat.view(n_points, 8)
            target_split = target_flat.view(n_points, 8)
            
            offset_loss = F.l1_loss(pred_split[:, :2], target_split[:, :2])
            height_loss = F.l1_loss(pred_split[:, 2], target_split[:, 2])
            dim_loss = F.l1_loss(pred_split[:, 3:6], target_split[:, 3:6])
            rot_loss = F.l1_loss(pred_split[:, 6:8], target_split[:, 6:8])
        else:
            offset_loss = torch.tensor(0.0, device=pred_heatmap.device)
            height_loss = torch.tensor(0.0, device=pred_heatmap.device)
            dim_loss = torch.tensor(0.0, device=pred_heatmap.device)
            rot_loss = torch.tensor(0.0, device=pred_heatmap.device)
        
        # Combine losses
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.offset_weight * offset_loss +
            self.height_weight * height_loss +
            self.dim_weight * dim_loss +
            self.rot_weight * rot_loss
        )
        
        return {
            "detection_total": total_loss,
            "detection_heatmap": heatmap_loss,
            "detection_offset": offset_loss,
            "detection_height": height_loss,
            "detection_dim": dim_loss,
            "detection_rot": rot_loss,
        }


class WaypointLoss(nn.Module):
    """
    Loss for waypoint prediction.
    
    Components:
    1. Position loss (L1 or Smooth-L1 for x, y)
    2. Heading loss (L1 or cosine for heading angle)
    
    Optional: curvature regularization for smooth trajectories
    """
    
    def __init__(
        self,
        position_weight: float = 1.0,
        heading_weight: float = 0.5,
        curvature_weight: float = 0.1,
        use_smooth_l1: bool = True,
    ):
        """
        Initialize waypoint loss.
        
        Args:
            position_weight: Weight for position (x, y) loss
            heading_weight: Weight for heading loss
            curvature_weight: Weight for curvature regularization
            use_smooth_l1: Whether to use smooth L1 (True) or L1 (False)
        """
        super().__init__()
        
        self.position_weight = position_weight
        self.heading_weight = heading_weight
        self.curvature_weight = curvature_weight
        self.use_smooth_l1 = use_smooth_l1
    
    def forward(
        self,
        pred_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
        waypoint_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute waypoint loss.
        
        Args:
            pred_waypoints: [B, K, 3] predicted waypoints (x, y, heading)
            target_waypoints: [B, K, 3] target waypoints
            waypoint_mask: [B, K] optional mask for valid waypoints
            
        Returns:
            Dict with loss components
        """
        if waypoint_mask is not None:
            # Apply mask
            mask = waypoint_mask.unsqueeze(-1).expand_as(pred_waypoints)
            pred_masked = pred_waypoints[mask].view(-1, 3)
            target_masked = target_waypoints[mask].view(-1, 3)
        else:
            pred_masked = pred_waypoints.reshape(-1, 3)
            target_masked = target_waypoints.reshape(-1, 3)
        
        # Position loss
        if self.use_smooth_l1:
            position_loss = F.smooth_l1_loss(
                pred_masked[:, :2], target_masked[:, :2]
            )
        else:
            position_loss = F.l1_loss(pred_masked[:, :2], target_masked[:, :2])
        
        # Heading loss (handle angle wraparound)
        pred_heading = pred_masked[:, 2]
        target_heading = target_masked[:, 2]
        
        # Use cosine similarity loss for angles
        heading_diff = torch.cos(pred_heading - target_heading)
        heading_loss = 1 - heading_diff.mean()  # 0 when aligned, 2 when opposite
        
        # Curvature regularization (penalize sharp turns)
        if self.curvature_weight > 0 and pred_waypoints.shape[1] > 2:
            # Compute curvature as second derivative of position
            dx = pred_waypoints[:, 1:, 0] - pred_waypoints[:, :-1, 0]
            dy = pred_waypoints[:, 1:, 1] - pred_waypoints[:, :-1, 1]
            
            ddx = dx[:, 1:] - dx[:, :-1]
            ddy = dy[:, 1:] - dy[:, :-1]
            
            curvature = (ddx ** 2 + ddy ** 2).mean()
        else:
            curvature = torch.tensor(0.0, device=pred_waypoints.device)
        
        # Combine losses
        total_loss = (
            self.position_weight * position_loss +
            self.heading_weight * heading_loss +
            self.curvature_weight * curvature
        )
        
        return {
            "waypoint_total": total_loss,
            "waypoint_position": position_loss,
            "waypoint_heading": heading_loss,
            "waypoint_curvature": curvature,
        }


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining detection and planning.
    
    Supports:
    1. Static weighting (fixed weights per task)
    2. Uncertainty weighting (learned task weights)
    """
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        planning_weight: float = 1.0,
        use_uncertainty_weighting: bool = False,
    ):
        """
        Initialize multi-task loss.
        
        Args:
            detection_weight: Weight for detection loss
            planning_weight: Weight for planning loss
            use_uncertainty_weighting: Whether to learn task weights
        """
        super().__init__()
        
        self.detection_loss = DetectionLoss()
        self.waypoint_loss = WaypointLoss()
        
        self.static_detection_weight = detection_weight
        self.static_planning_weight = planning_weight
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learned log variances for uncertainty weighting
            # Loss = 1/(2*sigma^2) * L + log(sigma)
            # = exp(-log_var) * L + log_var/2
            self.log_var_detection = nn.Parameter(torch.zeros(1))
            self.log_var_planning = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dict with model outputs:
                - heatmap: [B, C, H, W]
                - box_reg: [B, 8, H, W]
                - waypoints: [B, K, 3]
            targets: Dict with ground truth:
                - heatmap: [B, C, H, W]
                - box_reg: [B, 8, H, W]
                - reg_mask: [B, H, W]
                - waypoints: [B, K, 3]
                - waypoint_mask: [B, K] (optional)
                
        Returns:
            Dict with all loss components and total
        """
        losses = {}
        
        # Detection loss
        det_losses = self.detection_loss(
            pred_heatmap=predictions["heatmap"],
            pred_reg=predictions["box_reg"],
            target_heatmap=targets["heatmap"],
            target_reg=targets["box_reg"],
            target_mask=targets["reg_mask"],
        )
        losses.update(det_losses)
        
        # Planning loss
        plan_losses = self.waypoint_loss(
            pred_waypoints=predictions["waypoints"],
            target_waypoints=targets["waypoints"],
            waypoint_mask=targets.get("waypoint_mask"),
        )
        losses.update(plan_losses)
        
        # Combine with weighting
        if self.use_uncertainty_weighting:
            # Uncertainty weighting
            detection_weight = torch.exp(-self.log_var_detection)
            planning_weight = torch.exp(-self.log_var_planning)
            
            total_loss = (
                detection_weight * losses["detection_total"] + self.log_var_detection / 2 +
                planning_weight * losses["waypoint_total"] + self.log_var_planning / 2
            )
            
            losses["detection_weight"] = detection_weight.detach()
            losses["planning_weight"] = planning_weight.detach()
        else:
            total_loss = (
                self.static_detection_weight * losses["detection_total"] +
                self.static_planning_weight * losses["waypoint_total"]
            )
        
        losses["total"] = total_loss
        
        return losses


def create_detection_targets(
    boxes_3d: torch.Tensor,
    boxes_mask: torch.Tensor,
    heatmap_size: Tuple[int, int],
    bev_x_range: Tuple[float, float],
    bev_y_range: Tuple[float, float],
    bev_resolution: float,
    num_classes: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Create detection targets from 3D boxes.
    
    Args:
        boxes_3d: [B, M, 9] 3D boxes (x, y, z, l, w, h, yaw, class, track_id)
        boxes_mask: [B, M] valid box mask
        heatmap_size: (H, W) target heatmap size
        bev_x_range: (min, max) x range
        bev_y_range: (min, max) y range
        bev_resolution: BEV resolution in meters/pixel
        num_classes: Number of object classes
        
    Returns:
        Dict with:
        - heatmap: [B, C, H, W] target heatmap
        - box_reg: [B, 8, H, W] regression targets
        - reg_mask: [B, H, W] valid regression mask
    """
    from wod_fusion.models.detection_head import gaussian_radius, draw_gaussian
    
    batch_size = boxes_3d.shape[0]
    H, W = heatmap_size
    device = boxes_3d.device
    
    # Initialize targets
    heatmap = torch.zeros(batch_size, num_classes, H, W, device=device)
    box_reg = torch.zeros(batch_size, 8, H, W, device=device)
    reg_mask = torch.zeros(batch_size, H, W, device=device)
    
    for b in range(batch_size):
        valid_boxes = boxes_3d[b][boxes_mask[b]]
        
        for box in valid_boxes:
            x, y, z, l, w, h, yaw, cls, _ = box
            cls = int(cls.item())
            
            if cls < 0 or cls >= num_classes:
                continue
            
            # Convert to BEV coordinates
            x_bev = (x - bev_x_range[0]) / bev_resolution
            y_bev = (y - bev_y_range[0]) / bev_resolution
            
            # Check if in bounds
            if x_bev < 0 or x_bev >= W or y_bev < 0 or y_bev >= H:
                continue
            
            x_int, y_int = int(x_bev), int(y_bev)
            
            # Compute Gaussian radius
            radius = max(1, int(gaussian_radius((l / bev_resolution, w / bev_resolution))))
            
            # Draw Gaussian on heatmap
            draw_gaussian(heatmap[b, cls], (x_int, y_int), radius)
            
            # Set regression targets
            dx = x_bev - x_int
            dy = y_bev - y_int
            sin_yaw = torch.sin(yaw)
            cos_yaw = torch.cos(yaw)
            
            box_reg[b, 0, y_int, x_int] = dx
            box_reg[b, 1, y_int, x_int] = dy
            box_reg[b, 2, y_int, x_int] = z
            box_reg[b, 3, y_int, x_int] = l
            box_reg[b, 4, y_int, x_int] = w
            box_reg[b, 5, y_int, x_int] = h
            box_reg[b, 6, y_int, x_int] = sin_yaw
            box_reg[b, 7, y_int, x_int] = cos_yaw
            
            reg_mask[b, y_int, x_int] = 1
    
    return {
        "heatmap": heatmap,
        "box_reg": box_reg,
        "reg_mask": reg_mask,
    }
