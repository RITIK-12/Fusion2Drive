#!/usr/bin/env python3
"""
Demo script for visualizing Fusion2Drive predictions.

Usage:
    python scripts/demo.py --checkpoint checkpoints/best.pt --data_dir /path/to/waymo
    python scripts/demo.py --checkpoint checkpoints/best.pt --image_dir /path/to/images
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wod_fusion.models import FusionModel, FusionModelConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_model(checkpoint_path: str, device: str) -> FusionModel:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model_config = checkpoint.get("config", {})
    if isinstance(model_config, dict):
        model_config = FusionModelConfig(**model_config)
    
    model = FusionModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model


def visualize_predictions(
    image: np.ndarray,
    detections: dict,
    waypoints: np.ndarray,
    output_path: str,
):
    """Visualize predictions on image."""
    try:
        import cv2
    except ImportError:
        logging.warning("OpenCV not installed, skipping visualization")
        return
    
    # Clone image
    vis = image.copy()
    
    # Draw detections
    colors = {
        0: (0, 255, 0),    # Vehicle - green
        1: (255, 0, 0),    # Pedestrian - blue
        2: (0, 255, 255),  # Cyclist - yellow
    }
    
    if "boxes" in detections:
        boxes = detections["boxes"]
        labels = detections["labels"]
        scores = detections["scores"]
        
        # Project 3D boxes to 2D (simplified - just show centers)
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.3:
                continue
            
            # Get color
            color = colors.get(int(label), (255, 255, 255))
            
            # Draw center point (simplified visualization)
            # In practice, would project 3D box corners
            center_x = int(image.shape[1] / 2 + box[0] * 10)
            center_y = int(image.shape[0] - box[1] * 10)
            
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
            cv2.putText(
                vis,
                f"{score:.2f}",
                (center_x + 5, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    
    # Draw waypoints
    if waypoints is not None:
        waypoint_color = (0, 0, 255)  # Red
        
        prev_point = None
        for wp in waypoints:
            # Project to image (simplified)
            px = int(image.shape[1] / 2 + wp[0] * 10)
            py = int(image.shape[0] - wp[1] * 10)
            
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                cv2.circle(vis, (px, py), 3, waypoint_color, -1)
                
                if prev_point is not None:
                    cv2.line(vis, prev_point, (px, py), waypoint_color, 2)
                
                prev_point = (px, py)
    
    # Save
    cv2.imwrite(output_path, vis)
    logging.info(f"Saved visualization to {output_path}")


def run_demo_on_dataset(
    model: FusionModel,
    data_dir: str,
    output_dir: str,
    device: str,
    max_samples: int = 10,
):
    """Run demo on Waymo dataset."""
    from wod_fusion.data import WaymoDataset
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    dataset = WaymoDataset(
        data_dir=data_dir,
        split="validation",
        image_size=(256, 704),
        max_points=150000,
        augment=False,
    )
    
    logging.info(f"Running demo on {min(max_samples, len(dataset))} samples...")
    
    for idx in range(min(max_samples, len(dataset))):
        # Get sample
        sample = dataset[idx]
        
        # Prepare inputs
        images = sample["images"].unsqueeze(0).to(device)
        intrinsics = sample["intrinsics"].unsqueeze(0).to(device)
        extrinsics = sample["extrinsics"].unsqueeze(0).to(device)
        points = sample["points"].unsqueeze(0).to(device)
        points_mask = sample["points_mask"].unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model.inference(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                points=points,
                points_mask=points_mask,
            )
        
        # Get front camera image for visualization
        front_image = sample["images"][0].permute(1, 2, 0).numpy()
        front_image = (front_image * 255).astype(np.uint8)
        
        # Visualize
        visualize_predictions(
            image=front_image,
            detections={
                "boxes": outputs["detections"][0]["boxes"].cpu().numpy(),
                "labels": outputs["detections"][0]["labels"].cpu().numpy(),
                "scores": outputs["detections"][0]["scores"].cpu().numpy(),
            },
            waypoints=outputs["waypoints"][0].cpu().numpy(),
            output_path=str(output_dir / f"demo_{idx:04d}.png"),
        )
    
    logging.info(f"Saved {max_samples} visualizations to {output_dir}")


def run_demo_on_images(
    model: FusionModel,
    image_dir: str,
    output_dir: str,
    device: str,
):
    """Run demo on individual images."""
    try:
        import cv2
    except ImportError:
        logging.error("OpenCV required for image demo")
        return
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    logging.info(f"Found {len(image_paths)} images")
    logging.warning(
        "WARNING: Running inference from single images uses placeholder calibration data. "
        "Detection results will NOT be accurate. For proper evaluation, use the Waymo dataset "
        "with actual camera intrinsics/extrinsics and LiDAR data via --data_dir."
    )
    
    for image_path in image_paths:
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image, (704, 256))
        
        # Normalize
        image_tensor = torch.from_numpy(image_resized).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Stack for 5 cameras (placeholder - NOT valid for real inference)
        # Real inference requires all 5 camera views with proper calibration
        images = image_tensor.unsqueeze(0).unsqueeze(0).expand(1, 5, -1, -1, -1).to(device)
        
        # Placeholder calibration - NOT valid for real inference
        # Real inference requires actual camera intrinsics and extrinsics from the dataset
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 5, -1, -1).to(device)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(1, 5, -1, -1).to(device)
        
        # Empty LiDAR (camera-only mode for image demo)
        points = torch.zeros(1, 150000, 4).to(device)
        points_mask = torch.zeros(1, 150000, dtype=torch.bool).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model.inference(
                images=images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                points=points,
                points_mask=points_mask,
            )
        
        # Visualize
        output_path = output_dir / f"demo_{image_path.stem}.png"
        
        visualize_predictions(
            image=image_resized,
            detections={
                "boxes": outputs["detections"][0]["boxes"].cpu().numpy(),
                "labels": outputs["detections"][0]["labels"].cpu().numpy(),
                "scores": outputs["detections"][0]["scores"].cpu().numpy(),
            },
            waypoints=outputs["waypoints"][0].cpu().numpy(),
            output_path=str(output_path),
        )
    
    logging.info(f"Processed {len(image_paths)} images")


def main():
    parser = argparse.ArgumentParser(description="Fusion2Drive Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to Waymo dataset",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory with images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/demo",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use (-1 for CPU)",
    )
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Run demo
    if args.data_dir:
        run_demo_on_dataset(
            model=model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=device,
            max_samples=args.max_samples,
        )
    elif args.image_dir:
        run_demo_on_images(
            model=model,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            device=device,
        )
    else:
        logger.error("Please provide either --data_dir or --image_dir")
        sys.exit(1)
    
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()
