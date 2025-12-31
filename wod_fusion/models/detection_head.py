"""
CenterPoint-style 3D detection head.

Predicts:
- Heatmap: Object center locations per class
- Box regression: Size, rotation, velocity, etc.

Uses anchor-free detection similar to CenterNet/CenterPoint.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CenterPointHead(nn.Module):
    """
    CenterPoint-style anchor-free 3D detection head.
    
    Operates on BEV features and predicts:
    1. Heatmap: Per-class center point heatmap
    2. Box regression: For each center point:
       - offset: (dx, dy) sub-pixel offset
       - height: z center
       - dim: (length, width, height)
       - rot: (sin, cos) of heading angle
    
    Total regression channels: 2 + 1 + 3 + 2 = 8
    """
    
    # Class names for Waymo
    CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        head_channels: int = 64,
        heatmap_kernel: int = 3,
    ):
        """
        Initialize detection head.
        
        Args:
            in_channels: Input BEV feature channels
            num_classes: Number of object classes
            head_channels: Intermediate head channels
            heatmap_kernel: Kernel size for heatmap convolutions
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        
        # Heatmap head (one channel per class)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, heatmap_kernel, padding=heatmap_kernel // 2, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, 1),
        )
        
        # Box regression head
        # Channels: offset(2) + height(1) + dim(3) + rot(2) = 8
        self.reg_head = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 8, 1),
        )
        
        # Initialize heatmap bias for stable training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize heatmap bias to -2.19 (focal loss stable init)
        # This corresponds to ~0.1 initial probability
        for m in self.heatmap_head.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == self.num_classes:
                nn.init.constant_(m.bias, -2.19)
    
    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            bev_features: [B, C, H, W] BEV feature map
            
        Returns:
            Dict with:
            - heatmap: [B, num_classes, H, W] center heatmaps
            - box_reg: [B, 8, H, W] box regression
        """
        # Shared features
        shared = self.shared(bev_features)
        
        # Heads
        heatmap = self.heatmap_head(shared)
        box_reg = self.reg_head(shared)
        
        return {
            "heatmap": heatmap,
            "box_reg": box_reg,
        }
    
    def decode(
        self,
        heatmap: torch.Tensor,
        box_reg: torch.Tensor,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        bev_x_range: Tuple[float, float] = (-75.0, 75.0),
        bev_y_range: Tuple[float, float] = (-75.0, 75.0),
        bev_resolution: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode predictions to 3D boxes.
        
        Args:
            heatmap: [B, C, H, W] center heatmaps
            box_reg: [B, 8, H, W] box regression
            score_threshold: Minimum score to keep
            nms_threshold: NMS IoU threshold
            max_detections: Maximum detections per sample
            bev_x_range: BEV x range in meters
            bev_y_range: BEV y range in meters
            bev_resolution: BEV resolution in meters/pixel
            
        Returns:
            List of dicts per sample with:
            - boxes: [N, 7] 3D boxes (x, y, z, l, w, h, yaw)
            - scores: [N] confidence scores
            - labels: [N] class labels
        """
        batch_size = heatmap.shape[0]
        device = heatmap.device
        
        # Apply sigmoid to heatmap
        heatmap = torch.sigmoid(heatmap)
        
        # Simple peak extraction via max pooling
        heatmap_pool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        peaks = (heatmap == heatmap_pool) & (heatmap > score_threshold)
        
        results = []
        
        for b in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []
            
            for c in range(self.num_classes):
                # Get peak locations for this class
                peak_mask = peaks[b, c]
                peak_coords = torch.nonzero(peak_mask, as_tuple=False)  # [N, 2] (y, x)
                
                if peak_coords.shape[0] == 0:
                    continue
                
                # Get scores
                scores = heatmap[b, c, peak_coords[:, 0], peak_coords[:, 1]]
                
                # Get box regression values
                reg = box_reg[b, :, peak_coords[:, 0], peak_coords[:, 1]]  # [8, N]
                reg = reg.T  # [N, 8]
                
                # Decode boxes
                # reg: offset_x, offset_y, height, length, width, height_box, sin, cos
                offset = reg[:, :2]  # [N, 2]
                z = reg[:, 2]        # [N]
                dim = reg[:, 3:6]    # [N, 3] (l, w, h)
                rot = reg[:, 6:8]    # [N, 2] (sin, cos)
                
                # Compute center in BEV coordinates
                x_bev = peak_coords[:, 1].float() + offset[:, 0]
                y_bev = peak_coords[:, 0].float() + offset[:, 1]
                
                # Convert to meters
                x = x_bev * bev_resolution + bev_x_range[0]
                y = y_bev * bev_resolution + bev_y_range[0]
                
                # Compute heading from sin/cos
                yaw = torch.atan2(rot[:, 0], rot[:, 1])
                
                # Stack boxes
                boxes = torch.stack([x, y, z, dim[:, 0], dim[:, 1], dim[:, 2], yaw], dim=-1)
                
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(torch.full((boxes.shape[0],), c, device=device, dtype=torch.long))
            
            if len(boxes_list) > 0:
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
                all_labels = torch.cat(labels_list, dim=0)
                
                # Per-class NMS
                keep = self._nms_3d(all_boxes, all_scores, all_labels, nms_threshold)
                
                # Limit detections
                if keep.shape[0] > max_detections:
                    top_scores, top_idx = all_scores[keep].topk(max_detections)
                    keep = keep[top_idx]
                
                results.append({
                    "boxes": all_boxes[keep],
                    "scores": all_scores[keep],
                    "labels": all_labels[keep],
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 7, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, device=device, dtype=torch.long),
                })
        
        return results
    
    def _nms_3d(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        """
        Simple BEV NMS (approximating 3D IoU with 2D).
        
        Args:
            boxes: [N, 7] 3D boxes
            scores: [N] scores
            labels: [N] class labels
            threshold: IoU threshold
            
        Returns:
            [K] indices to keep
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        
        # Sort by score
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
            
            # Compute BEV IoU with remaining boxes
            remaining = order[1:]
            
            # Simple BEV rectangle IoU (ignoring rotation for speed)
            box_i = boxes[i]
            boxes_j = boxes[remaining]
            
            # Compute intersection
            x1 = torch.max(box_i[0] - box_i[3] / 2, boxes_j[:, 0] - boxes_j[:, 3] / 2)
            y1 = torch.max(box_i[1] - box_i[4] / 2, boxes_j[:, 1] - boxes_j[:, 4] / 2)
            x2 = torch.min(box_i[0] + box_i[3] / 2, boxes_j[:, 0] + boxes_j[:, 3] / 2)
            y2 = torch.min(box_i[1] + box_i[4] / 2, boxes_j[:, 1] + boxes_j[:, 4] / 2)
            
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_i = box_i[3] * box_i[4]
            area_j = boxes_j[:, 3] * boxes_j[:, 4]
            union = area_i + area_j - inter
            iou = inter / (union + 1e-6)
            
            # Keep boxes with low IoU or different class
            keep_mask = (iou < threshold) | (labels[remaining] != labels[i])
            order = remaining[keep_mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.5) -> float:
    """
    Compute Gaussian radius for heatmap encoding.
    
    Args:
        det_size: (height, width) of detection
        min_overlap: Minimum IoU overlap
        
    Returns:
        Gaussian radius
    """
    height, width = det_size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    
    return min(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int, k: float = 1.0):
    """
    Draw a Gaussian on heatmap at given center.
    
    Args:
        heatmap: [H, W] heatmap tensor
        center: (x, y) center coordinates
        radius: Gaussian radius
        k: Peak value
    """
    diameter = 2 * radius + 1
    gaussian = torch.exp(
        -torch.arange(-radius, radius + 1, device=heatmap.device).float() ** 2 / (2 * (radius / 3) ** 2)
    )
    gaussian = gaussian.unsqueeze(0) * gaussian.unsqueeze(1)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape
    
    left = min(x, radius)
    right = min(width - x, radius + 1)
    top = min(y, radius)
    bottom = min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
