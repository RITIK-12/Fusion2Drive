"""
Evaluation metrics for 3D detection and planning.

Implements:
- mAP and mAPH for 3D detection (Waymo-style)
- ADE and FDE for trajectory prediction
- Collision proxy metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ============================================================================
# 3D Detection Metrics
# ============================================================================

@dataclass
class DetectionResult:
    """Single detection result."""
    boxes: np.ndarray  # [N, 7] (x, y, z, l, w, h, yaw)
    scores: np.ndarray  # [N]
    labels: np.ndarray  # [N]


@dataclass
class GroundTruth:
    """Ground truth for a frame."""
    boxes: np.ndarray  # [M, 7]
    labels: np.ndarray  # [M]
    difficulty: Optional[np.ndarray] = None  # [M] (1=LEVEL_1, 2=LEVEL_2)


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 3D IoU between two boxes.
    
    Simplified: Uses BEV IoU * height overlap ratio.
    For exact 3D IoU, would need proper rotated box intersection.
    
    Args:
        box1, box2: [7] boxes (x, y, z, l, w, h, yaw)
        
    Returns:
        IoU value in [0, 1]
    """
    # BEV IoU (ignoring rotation for simplicity)
    x1, y1, z1, l1, w1, h1, _ = box1
    x2, y2, z2, l2, w2, h2, _ = box2
    
    # BEV rectangle intersection
    x1_min, x1_max = x1 - l1/2, x1 + l1/2
    y1_min, y1_max = y1 - w1/2, y1 + w1/2
    x2_min, x2_max = x2 - l2/2, x2 + l2/2
    y2_min, y2_max = y2 - w2/2, y2 + w2/2
    
    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_bev = inter_x * inter_y
    
    area1_bev = l1 * w1
    area2_bev = l2 * w2
    union_bev = area1_bev + area2_bev - inter_bev
    
    iou_bev = inter_bev / (union_bev + 1e-6)
    
    # Height overlap
    z1_min, z1_max = z1 - h1/2, z1 + h1/2
    z2_min, z2_max = z2 - h2/2, z2 + h2/2
    
    inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    union_z = max(z1_max, z2_max) - min(z1_min, z2_min)
    
    height_ratio = inter_z / (union_z + 1e-6)
    
    # Approximate 3D IoU
    iou_3d = iou_bev * height_ratio
    
    return iou_3d


def compute_heading_accuracy(pred_yaw: float, gt_yaw: float) -> float:
    """
    Compute heading accuracy factor for mAPH.
    
    Returns value in [0, 1] based on heading difference.
    """
    diff = abs(pred_yaw - gt_yaw)
    # Normalize to [0, pi]
    diff = min(diff, 2 * np.pi - diff)
    
    # Convert to accuracy (1.0 at 0 difference, 0.0 at pi/2+)
    accuracy = max(0, 1 - diff / (np.pi / 2))
    
    return accuracy


def compute_ap(
    pred_scores: List[float],
    pred_matched: List[bool],
    num_gt: int,
    heading_accuracies: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Compute Average Precision (AP) and Average Precision with Heading (APH).
    
    Args:
        pred_scores: Confidence scores for each prediction
        pred_matched: Whether each prediction matched a GT box
        num_gt: Total number of ground truth boxes
        heading_accuracies: Heading accuracy for each matched prediction
        
    Returns:
        (AP, APH)
    """
    if num_gt == 0:
        return 0.0, 0.0
    
    # Sort by score
    sorted_indices = np.argsort(pred_scores)[::-1]
    
    tp = np.zeros(len(pred_scores))
    tp_weighted = np.zeros(len(pred_scores))
    
    for i, idx in enumerate(sorted_indices):
        if pred_matched[idx]:
            tp[i] = 1
            if heading_accuracies is not None:
                tp_weighted[i] = heading_accuracies[idx]
            else:
                tp_weighted[i] = 1
    
    # Cumulative sum
    tp_cumsum = np.cumsum(tp)
    tp_weighted_cumsum = np.cumsum(tp_weighted)
    
    # Precision and recall
    precision = tp_cumsum / (np.arange(len(tp)) + 1)
    recall = tp_cumsum / num_gt
    
    precision_weighted = tp_weighted_cumsum / (np.arange(len(tp)) + 1)
    recall_weighted = tp_weighted_cumsum / num_gt
    
    # AP (area under precision-recall curve)
    # Use 11-point interpolation
    ap = 0.0
    aph = 0.0
    
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max() / 11
        
        mask_w = recall_weighted >= t
        if mask_w.any():
            aph += precision_weighted[mask_w].max() / 11
    
    return ap, aph


def compute_detection_metrics(
    predictions: List[DetectionResult],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.7,
    class_names: List[str] = None,
) -> Dict[str, float]:
    """
    Compute 3D detection metrics (mAP, mAPH).
    
    Metrics are computed:
    - Per class
    - By difficulty level (LEVEL_1, LEVEL_2) if available
    - Overall
    
    Args:
        predictions: List of detection results per frame
        ground_truths: List of ground truths per frame
        iou_threshold: IoU threshold for matching
        class_names: Names of classes
        
    Returns:
        Dict with metric values
    """
    if class_names is None:
        class_names = ["vehicle", "pedestrian", "cyclist"]
    
    num_classes = len(class_names)
    
    # Collect all predictions and matches per class
    class_predictions = {c: {"scores": [], "matched": [], "heading_acc": []} for c in range(num_classes)}
    class_num_gt = {c: 0 for c in range(num_classes)}
    
    for pred, gt in zip(predictions, ground_truths):
        for c in range(num_classes):
            # Get predictions and GT for this class
            pred_mask = pred.labels == c
            gt_mask = gt.labels == c
            
            pred_boxes = pred.boxes[pred_mask]
            pred_scores = pred.scores[pred_mask]
            gt_boxes = gt.boxes[gt_mask]
            
            class_num_gt[c] += len(gt_boxes)
            
            # Match predictions to GT
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            
            # Sort predictions by score
            score_order = np.argsort(pred_scores)[::-1]
            
            for idx in score_order:
                pred_box = pred_boxes[idx]
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = compute_iou_3d(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                matched = best_iou >= iou_threshold and best_gt_idx >= 0
                
                class_predictions[c]["scores"].append(pred_scores[idx])
                class_predictions[c]["matched"].append(matched)
                
                if matched:
                    gt_matched[best_gt_idx] = True
                    heading_acc = compute_heading_accuracy(
                        pred_box[6], gt_boxes[best_gt_idx][6]
                    )
                    class_predictions[c]["heading_acc"].append(heading_acc)
                else:
                    class_predictions[c]["heading_acc"].append(0.0)
    
    # Compute AP per class
    metrics = {}
    
    for c in range(num_classes):
        ap, aph = compute_ap(
            class_predictions[c]["scores"],
            class_predictions[c]["matched"],
            class_num_gt[c],
            class_predictions[c]["heading_acc"],
        )
        
        metrics[f"AP/{class_names[c]}"] = ap
        metrics[f"APH/{class_names[c]}"] = aph
    
    # Mean AP
    metrics["mAP"] = np.mean([metrics[f"AP/{name}"] for name in class_names])
    metrics["mAPH"] = np.mean([metrics[f"APH/{name}"] for name in class_names])
    
    return metrics


# ============================================================================
# Planning Metrics
# ============================================================================

def compute_ade(
    pred_waypoints: np.ndarray,
    gt_waypoints: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Average Displacement Error (ADE).
    
    ADE = mean euclidean distance across all waypoints
    
    Args:
        pred_waypoints: [K, 2] or [K, 3] predicted waypoints
        gt_waypoints: [K, 2] or [K, 3] ground truth waypoints
        mask: [K] optional mask for valid waypoints
        
    Returns:
        ADE in meters
    """
    # Use only x, y for displacement
    pred_xy = pred_waypoints[:, :2]
    gt_xy = gt_waypoints[:, :2]
    
    distances = np.linalg.norm(pred_xy - gt_xy, axis=1)
    
    if mask is not None:
        distances = distances[mask]
    
    return distances.mean()


def compute_fde(
    pred_waypoints: np.ndarray,
    gt_waypoints: np.ndarray,
    horizon_idx: int = -1,
) -> float:
    """
    Compute Final Displacement Error (FDE).
    
    FDE = euclidean distance at final waypoint
    
    Args:
        pred_waypoints: [K, 2] or [K, 3] predicted waypoints
        gt_waypoints: [K, 2] or [K, 3] ground truth waypoints
        horizon_idx: Index of waypoint to use (default: last)
        
    Returns:
        FDE in meters
    """
    pred_xy = pred_waypoints[horizon_idx, :2]
    gt_xy = gt_waypoints[horizon_idx, :2]
    
    return np.linalg.norm(pred_xy - gt_xy)


def compute_heading_error(
    pred_waypoints: np.ndarray,
    gt_waypoints: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute average heading error.
    
    Args:
        pred_waypoints: [K, 3] with (x, y, heading)
        gt_waypoints: [K, 3] with (x, y, heading)
        mask: [K] optional mask
        
    Returns:
        Mean heading error in radians
    """
    if pred_waypoints.shape[1] < 3:
        return 0.0
    
    pred_heading = pred_waypoints[:, 2]
    gt_heading = gt_waypoints[:, 2]
    
    # Handle angle wraparound
    diff = np.abs(pred_heading - gt_heading)
    diff = np.minimum(diff, 2 * np.pi - diff)
    
    if mask is not None:
        diff = diff[mask]
    
    return diff.mean()


def compute_planning_metrics(
    pred_waypoints: np.ndarray,
    gt_waypoints: np.ndarray,
    horizons: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute all planning metrics.
    
    Args:
        pred_waypoints: [B, K, 3] predicted waypoints
        gt_waypoints: [B, K, 3] ground truth waypoints
        horizons: Time horizons for each waypoint (for reporting)
        
    Returns:
        Dict with metrics
    """
    if horizons is None:
        horizons = [0.5 * (i + 1) for i in range(pred_waypoints.shape[1])]
    
    batch_size = pred_waypoints.shape[0]
    
    all_ade = []
    all_fde = []
    all_heading_error = []
    
    for b in range(batch_size):
        ade = compute_ade(pred_waypoints[b], gt_waypoints[b])
        fde = compute_fde(pred_waypoints[b], gt_waypoints[b])
        he = compute_heading_error(pred_waypoints[b], gt_waypoints[b])
        
        all_ade.append(ade)
        all_fde.append(fde)
        all_heading_error.append(he)
    
    metrics = {
        "ADE": np.mean(all_ade),
        "FDE": np.mean(all_fde),
        "heading_error": np.mean(all_heading_error),
    }
    
    # Add per-horizon metrics
    for i, h in enumerate(horizons[:3]):  # First 3 horizons
        horizon_ade = []
        for b in range(batch_size):
            dist = np.linalg.norm(
                pred_waypoints[b, i, :2] - gt_waypoints[b, i, :2]
            )
            horizon_ade.append(dist)
        metrics[f"ADE@{h:.1f}s"] = np.mean(horizon_ade)
    
    return metrics


# ============================================================================
# Collision Proxy Metrics
# ============================================================================

def compute_collision_proxy(
    pred_waypoints: np.ndarray,
    detected_boxes: np.ndarray,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    safety_margin: float = 0.5,
) -> Dict[str, float]:
    """
    Compute collision proxy metrics.
    
    Checks if predicted ego trajectory intersects with detected objects.
    This is an open-loop approximation - objects are assumed static.
    
    Args:
        pred_waypoints: [K, 3] predicted waypoints (x, y, heading)
        detected_boxes: [N, 7] detected boxes (x, y, z, l, w, h, yaw)
        ego_length: Ego vehicle length
        ego_width: Ego vehicle width
        safety_margin: Additional safety margin
        
    Returns:
        Dict with collision metrics
    """
    if len(detected_boxes) == 0:
        return {
            "collision_rate": 0.0,
            "min_distance": float("inf"),
            "collision_count": 0,
        }
    
    num_waypoints = len(pred_waypoints)
    collision_count = 0
    min_distance = float("inf")
    
    for wp in pred_waypoints:
        ego_x, ego_y = wp[0], wp[1]
        
        # Simple collision check: center distance < sum of half-dimensions + margin
        for box in detected_boxes:
            obj_x, obj_y = box[0], box[1]
            obj_l, obj_w = box[3], box[4]
            
            dist = np.sqrt((ego_x - obj_x) ** 2 + (ego_y - obj_y) ** 2)
            
            # Approximate safe distance (ignoring rotation)
            safe_dist = (ego_length + obj_l) / 2 + (ego_width + obj_w) / 4 + safety_margin
            
            min_distance = min(min_distance, dist)
            
            if dist < safe_dist:
                collision_count += 1
                break
    
    return {
        "collision_rate": collision_count / num_waypoints,
        "min_distance": min_distance if min_distance != float("inf") else 0.0,
        "collision_count": collision_count,
    }


def compute_all_collision_metrics(
    pred_waypoints: np.ndarray,
    detections: List[Dict],
) -> Dict[str, float]:
    """
    Compute collision metrics over a batch.
    
    Args:
        pred_waypoints: [B, K, 3] predicted waypoints
        detections: List of detection dicts per sample
        
    Returns:
        Dict with aggregated collision metrics
    """
    batch_size = pred_waypoints.shape[0]
    
    all_collision_rates = []
    all_min_distances = []
    total_collisions = 0
    
    for b in range(batch_size):
        boxes = detections[b]["boxes"].numpy() if torch.is_tensor(detections[b]["boxes"]) else detections[b]["boxes"]
        
        metrics = compute_collision_proxy(pred_waypoints[b], boxes)
        
        all_collision_rates.append(metrics["collision_rate"])
        all_min_distances.append(metrics["min_distance"])
        total_collisions += metrics["collision_count"]
    
    return {
        "collision_rate": np.mean(all_collision_rates),
        "avg_min_distance": np.mean(all_min_distances),
        "total_collisions": total_collisions,
    }
