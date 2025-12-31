"""
Evaluator for running full evaluation pipeline.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wod_fusion.models import FusionModel
from wod_fusion.eval.metrics import (
    compute_detection_metrics,
    compute_planning_metrics,
    compute_all_collision_metrics,
    DetectionResult,
    GroundTruth,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluation pipeline for Fusion2Drive model.
    
    Runs inference on a dataset and computes:
    - 3D detection metrics (mAP, mAPH)
    - Planning metrics (ADE, FDE)
    - Collision proxy metrics
    - Runtime metrics
    """
    
    CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]
    
    def __init__(
        self,
        model: FusionModel,
        dataloader: DataLoader,
        device: str = "cuda",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained FusionModel
            dataloader: Evaluation data loader
            device: Device for inference
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/eval")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(
        self,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        compute_runtime: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Args:
            score_threshold: Detection score threshold
            nms_threshold: NMS IoU threshold
            compute_runtime: Whether to measure inference latency
            
        Returns:
            Dict with all metrics
        """
        logger.info("Starting evaluation...")
        
        all_predictions = []
        all_ground_truths = []
        all_pred_waypoints = []
        all_gt_waypoints = []
        all_detections = []
        
        latencies = []
        
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            # Move to device
            batch = self._to_device(batch)
            
            # Measure latency
            if compute_runtime:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
            
            # Inference
            outputs = self.model.inference(
                images=batch["images"],
                intrinsics=batch["intrinsics"],
                extrinsics=batch["extrinsics"],
                points=batch["points"],
                points_mask=batch["points_mask"],
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
            )
            
            # Measure latency
            if compute_runtime:
                if self.device.type == "cuda":
                    end_time.record()
                    torch.cuda.synchronize()
                    latency = start_time.elapsed_time(end_time)  # ms
                    latencies.append(latency / batch["images"].shape[0])  # Per-sample
            
            # Collect predictions
            batch_size = batch["images"].shape[0]
            
            for b in range(batch_size):
                # Detection results
                det_result = DetectionResult(
                    boxes=outputs["detections"][b]["boxes"].cpu().numpy(),
                    scores=outputs["detections"][b]["scores"].cpu().numpy(),
                    labels=outputs["detections"][b]["labels"].cpu().numpy(),
                )
                all_predictions.append(det_result)
                
                # Ground truth
                valid_mask = batch["boxes_mask"][b].cpu().numpy()
                gt_boxes = batch["boxes_3d"][b][valid_mask].cpu().numpy()
                
                gt = GroundTruth(
                    boxes=gt_boxes[:, :7] if len(gt_boxes) > 0 else np.zeros((0, 7)),
                    labels=gt_boxes[:, 7].astype(int) if len(gt_boxes) > 0 else np.zeros(0, dtype=int),
                )
                all_ground_truths.append(gt)
                
                # Waypoints
                all_pred_waypoints.append(outputs["waypoints"][b].cpu().numpy())
                all_gt_waypoints.append(batch["waypoints"][b].cpu().numpy())
                
                # Detections for collision check
                all_detections.append(outputs["detections"][b])
        
        # Stack waypoints
        pred_waypoints = np.stack(all_pred_waypoints, axis=0)
        gt_waypoints = np.stack(all_gt_waypoints, axis=0)
        
        # Compute metrics
        metrics = {}
        
        # Detection metrics
        logger.info("Computing detection metrics...")
        det_metrics = compute_detection_metrics(
            all_predictions,
            all_ground_truths,
            iou_threshold=0.7,
            class_names=self.CLASS_NAMES,
        )
        metrics.update(det_metrics)
        
        # Also compute at lower IoU threshold
        det_metrics_05 = compute_detection_metrics(
            all_predictions,
            all_ground_truths,
            iou_threshold=0.5,
            class_names=self.CLASS_NAMES,
        )
        metrics["mAP@0.5"] = det_metrics_05["mAP"]
        metrics["mAPH@0.5"] = det_metrics_05["mAPH"]
        
        # Planning metrics
        logger.info("Computing planning metrics...")
        plan_metrics = compute_planning_metrics(pred_waypoints, gt_waypoints)
        metrics.update(plan_metrics)
        
        # Collision metrics
        logger.info("Computing collision metrics...")
        collision_metrics = compute_all_collision_metrics(pred_waypoints, all_detections)
        metrics.update(collision_metrics)
        
        # Runtime metrics
        if compute_runtime and latencies:
            metrics["latency_mean_ms"] = np.mean(latencies)
            metrics["latency_std_ms"] = np.std(latencies)
            metrics["fps"] = 1000 / np.mean(latencies)
        
        # Log summary
        logger.info("=" * 50)
        logger.info("Evaluation Results:")
        logger.info("-" * 50)
        logger.info(f"Detection:")
        logger.info(f"  mAP@0.7: {metrics['mAP']:.4f}")
        logger.info(f"  mAPH@0.7: {metrics['mAPH']:.4f}")
        for name in self.CLASS_NAMES:
            logger.info(f"  AP/{name}: {metrics[f'AP/{name}']:.4f}")
        logger.info(f"Planning:")
        logger.info(f"  ADE: {metrics['ADE']:.4f} m")
        logger.info(f"  FDE: {metrics['FDE']:.4f} m")
        logger.info(f"Collision:")
        logger.info(f"  Collision Rate: {metrics['collision_rate']:.4f}")
        if compute_runtime and latencies:
            logger.info(f"Runtime:")
            logger.info(f"  Latency: {metrics['latency_mean_ms']:.2f} ± {metrics['latency_std_ms']:.2f} ms")
            logger.info(f"  FPS: {metrics['fps']:.1f}")
        logger.info("=" * 50)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result
    
    def _save_results(self, metrics: Dict):
        """Save evaluation results."""
        # Save as JSON
        json_path = self.output_dir / "metrics.json"
        with open(json_path, "w") as f:
            # Convert numpy types to Python types
            serializable = {}
            for k, v in metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    serializable[k] = float(v)
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Saved metrics to {json_path}")
        
        # Save as text summary
        summary_path = self.output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("Fusion2Drive Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("3D Detection Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"mAP@0.7: {metrics['mAP']:.4f}\n")
            f.write(f"mAPH@0.7: {metrics['mAPH']:.4f}\n")
            for name in self.CLASS_NAMES:
                f.write(f"AP/{name}: {metrics[f'AP/{name}']:.4f}\n")
            
            f.write("\nPlanning Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"ADE: {metrics['ADE']:.4f} m\n")
            f.write(f"FDE: {metrics['FDE']:.4f} m\n")
            f.write(f"Heading Error: {metrics['heading_error']:.4f} rad\n")
            
            f.write("\nCollision Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Collision Rate: {metrics['collision_rate']:.4f}\n")
            f.write(f"Avg Min Distance: {metrics['avg_min_distance']:.4f} m\n")
            
            if "latency_mean_ms" in metrics:
                f.write("\nRuntime Metrics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Latency: {metrics['latency_mean_ms']:.2f} ms\n")
                f.write(f"FPS: {metrics['fps']:.1f}\n")
        
        logger.info(f"Saved summary to {summary_path}")
    
    def run_ablation(
        self,
        score_threshold: float = 0.3,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study comparing different modality configurations.
        
        Returns:
            Dict mapping config name to metrics
        """
        results = {}
        
        # Store original config
        model = self.model.module if hasattr(self.model, "module") else self.model
        orig_use_lidar = model.config.use_lidar
        orig_use_camera = model.config.use_camera
        
        configs = [
            ("fusion", True, True),
            ("lidar_only", True, False),
            ("camera_only", False, True),
        ]
        
        for name, use_lidar, use_camera in configs:
            if not use_lidar and model.lidar_encoder is None:
                continue
            if not use_camera and model.camera_encoder is None:
                continue
            
            logger.info(f"Running ablation: {name}")
            
            # Update config (this is a simplified approach)
            # In practice, would need separate model instances
            model.config.use_lidar = use_lidar
            model.config.use_camera = use_camera
            
            # Run evaluation
            metrics = self.evaluate(score_threshold=score_threshold, compute_runtime=False)
            results[name] = metrics
        
        # Restore original config
        model.config.use_lidar = orig_use_lidar
        model.config.use_camera = orig_use_camera
        
        # Save ablation results
        ablation_path = self.output_dir / "ablation.json"
        with open(ablation_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        
        logger.info(f"Saved ablation results to {ablation_path}")
        
        return results
