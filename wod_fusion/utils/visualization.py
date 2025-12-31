"""
Visualization utilities for Fusion2Drive.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# Color palettes
CLASS_COLORS = {
    0: (0, 255, 0),      # Vehicle - green
    1: (0, 0, 255),      # Pedestrian - blue
    2: (255, 255, 0),    # Cyclist - yellow
    "vehicle": (0, 255, 0),
    "pedestrian": (0, 0, 255),
    "cyclist": (255, 255, 0),
}

WAYPOINT_COLOR = (255, 0, 0)  # Red
GT_WAYPOINT_COLOR = (0, 255, 255)  # Cyan
EGO_COLOR = (255, 165, 0)  # Orange


def draw_box_3d_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw 3D box on image from projected corners.
    
    Args:
        image: Image to draw on (will be modified in place)
        corners_2d: (8, 2) projected corner coordinates
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        Image with box drawn
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, skipping visualization")
        return image
    
    corners = corners_2d.astype(int)
    
    # Draw bottom face
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, thickness)
    
    # Draw top face
    for i in range(4, 8):
        cv2.line(image, tuple(corners[i]), tuple(corners[4 + (i - 3) % 4]), color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[i + 4]), color, thickness)
    
    return image


def draw_detections_on_bev(
    bev_image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    bev_range: Tuple[float, float, float, float] = (-50, -50, 50, 50),
    score_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw 3D detection boxes on BEV image.
    
    Args:
        bev_image: BEV image to draw on
        boxes: (N, 7) boxes [x, y, z, l, w, h, heading]
        labels: (N,) class labels
        scores: (N,) confidence scores
        bev_range: (x_min, y_min, x_max, y_max) in meters
        score_threshold: Minimum score to draw
        
    Returns:
        BEV image with boxes drawn
    """
    try:
        import cv2
    except ImportError:
        return bev_image
    
    h, w = bev_image.shape[:2]
    x_min, y_min, x_max, y_max = bev_range
    
    # Scale factors
    scale_x = w / (x_max - x_min)
    scale_y = h / (y_max - y_min)
    
    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        x, y, _, length, width, _, heading = box
        label = int(labels[i])
        
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        # Get corners in world coordinates
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        corners = np.array([
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ])
        
        # Rotate
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = corners @ rotation.T
        
        # Translate
        corners += np.array([x, y])
        
        # Convert to pixel coordinates
        corners_px = np.zeros_like(corners, dtype=int)
        corners_px[:, 0] = ((corners[:, 0] - x_min) * scale_x).astype(int)
        corners_px[:, 1] = ((y_max - corners[:, 1]) * scale_y).astype(int)  # Flip Y
        
        # Draw box
        for j in range(4):
            pt1 = tuple(corners_px[j])
            pt2 = tuple(corners_px[(j + 1) % 4])
            cv2.line(bev_image, pt1, pt2, color, 2)
        
        # Draw heading indicator
        front_center = (corners_px[0] + corners_px[1]) // 2
        cv2.circle(bev_image, tuple(front_center), 3, color, -1)
    
    return bev_image


def draw_waypoints_on_bev(
    bev_image: np.ndarray,
    waypoints: np.ndarray,
    bev_range: Tuple[float, float, float, float] = (-50, -50, 50, 50),
    color: Tuple[int, int, int] = WAYPOINT_COLOR,
    radius: int = 4,
    draw_lines: bool = True,
) -> np.ndarray:
    """
    Draw waypoints on BEV image.
    
    Args:
        bev_image: BEV image to draw on
        waypoints: (T, 2) or (T, 3) waypoints in world frame
        bev_range: BEV coordinate range
        color: Color for waypoints
        radius: Circle radius
        draw_lines: Whether to connect waypoints with lines
        
    Returns:
        BEV image with waypoints drawn
    """
    try:
        import cv2
    except ImportError:
        return bev_image
    
    h, w = bev_image.shape[:2]
    x_min, y_min, x_max, y_max = bev_range
    
    scale_x = w / (x_max - x_min)
    scale_y = h / (y_max - y_min)
    
    # Convert to pixel coordinates
    waypoints_px = np.zeros((len(waypoints), 2), dtype=int)
    waypoints_px[:, 0] = ((waypoints[:, 0] - x_min) * scale_x).astype(int)
    waypoints_px[:, 1] = ((y_max - waypoints[:, 1]) * scale_y).astype(int)
    
    # Draw lines
    if draw_lines:
        for i in range(len(waypoints_px) - 1):
            pt1 = tuple(waypoints_px[i])
            pt2 = tuple(waypoints_px[i + 1])
            cv2.line(bev_image, pt1, pt2, color, 2)
    
    # Draw points
    for i, pt in enumerate(waypoints_px):
        # Fade color over time
        alpha = 1.0 - (i / len(waypoints_px)) * 0.5
        faded_color = tuple(int(c * alpha) for c in color)
        cv2.circle(bev_image, tuple(pt), radius, faded_color, -1)
    
    return bev_image


def create_bev_canvas(
    size: Tuple[int, int] = (800, 800),
    bev_range: Tuple[float, float, float, float] = (-50, -50, 50, 50),
    grid_spacing: float = 10.0,
) -> np.ndarray:
    """
    Create a BEV canvas with grid lines.
    
    Args:
        size: Image size (height, width)
        bev_range: Coordinate range
        grid_spacing: Grid line spacing in meters
        
    Returns:
        BEV canvas image
    """
    try:
        import cv2
    except ImportError:
        return np.zeros((*size, 3), dtype=np.uint8)
    
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    x_min, y_min, x_max, y_max = bev_range
    scale_x = w / (x_max - x_min)
    scale_y = h / (y_max - y_min)
    
    # Draw grid
    grid_color = (50, 50, 50)
    
    # Vertical lines
    for x in np.arange(np.ceil(x_min / grid_spacing) * grid_spacing, x_max, grid_spacing):
        px = int((x - x_min) * scale_x)
        cv2.line(canvas, (px, 0), (px, h), grid_color, 1)
    
    # Horizontal lines
    for y in np.arange(np.ceil(y_min / grid_spacing) * grid_spacing, y_max, grid_spacing):
        py = int((y_max - y) * scale_y)
        cv2.line(canvas, (0, py), (w, py), grid_color, 1)
    
    # Draw axes
    center_x = int((0 - x_min) * scale_x)
    center_y = int((y_max - 0) * scale_y)
    
    # X axis (red)
    cv2.line(canvas, (center_x, center_y), (w, center_y), (0, 0, 150), 1)
    
    # Y axis (green)
    cv2.line(canvas, (center_x, center_y), (center_x, 0), (0, 150, 0), 1)
    
    # Draw ego vehicle position
    cv2.circle(canvas, (center_x, center_y), 8, EGO_COLOR, -1)
    
    return canvas


def visualize_predictions(
    images: np.ndarray,
    detections: Dict[str, np.ndarray],
    waypoints: np.ndarray,
    gt_detections: Optional[Dict[str, np.ndarray]] = None,
    gt_waypoints: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create full visualization of predictions.
    
    Args:
        images: (N, H, W, 3) camera images
        detections: Dict with 'boxes', 'labels', 'scores'
        waypoints: (T, 2) or (T, 3) predicted waypoints
        gt_detections: Optional ground truth detections
        gt_waypoints: Optional ground truth waypoints
        output_path: Path to save visualization
        
    Returns:
        Visualization image
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed")
        return None
    
    # Create BEV canvas
    bev_canvas = create_bev_canvas()
    
    # Draw detections on BEV
    if "boxes" in detections:
        draw_detections_on_bev(
            bev_canvas,
            detections["boxes"],
            detections["labels"],
            detections.get("scores"),
        )
    
    # Draw ground truth detections
    if gt_detections is not None and "boxes" in gt_detections:
        # Draw GT in different style (dashed or different color)
        draw_detections_on_bev(
            bev_canvas,
            gt_detections["boxes"],
            gt_detections["labels"],
            score_threshold=0.0,
        )
    
    # Draw ground truth waypoints
    if gt_waypoints is not None:
        draw_waypoints_on_bev(
            bev_canvas,
            gt_waypoints,
            color=GT_WAYPOINT_COLOR,
        )
    
    # Draw predicted waypoints
    draw_waypoints_on_bev(bev_canvas, waypoints)
    
    # Create camera image mosaic
    if images is not None and len(images) > 0:
        # Resize images for display
        display_h, display_w = 200, 350
        resized = []
        for img in images[:5]:  # Max 5 cameras
            resized.append(cv2.resize(img, (display_w, display_h)))
        
        # Arrange in grid
        if len(resized) <= 3:
            camera_mosaic = np.hstack(resized)
        else:
            top_row = np.hstack(resized[:3])
            bottom_row = np.hstack(resized[3:] + [np.zeros((display_h, display_w, 3), dtype=np.uint8)] * (3 - len(resized[3:])))
            camera_mosaic = np.vstack([top_row, bottom_row])
        
        # Combine with BEV
        bev_resized = cv2.resize(bev_canvas, (camera_mosaic.shape[1], camera_mosaic.shape[1]))
        visualization = np.vstack([camera_mosaic, bev_resized])
    else:
        visualization = bev_canvas
    
    # Save if path provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved visualization to {output_path}")
    
    return visualization


def save_video_from_predictions(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 10.0,
):
    """
    Save a list of visualization frames as video.
    
    Args:
        frames: List of visualization images
        output_path: Output video path
        fps: Video frame rate
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV required for video export")
        return
    
    if not frames:
        logger.warning("No frames to save")
        return
    
    h, w = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    logger.info(f"Saved video with {len(frames)} frames to {output_path}")


def plot_training_curves(
    metrics: Dict[str, List[float]],
    output_path: Optional[str] = None,
) -> Any:
    """
    Plot training curves.
    
    Args:
        metrics: Dict mapping metric name to list of values per epoch
        output_path: Path to save plot
        
    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting")
        return None
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(4 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved training curves to {output_path}")
    
    return fig
