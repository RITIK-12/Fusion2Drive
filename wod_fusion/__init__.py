"""
Fusion2Drive: Waymo Perception Fusion to Ego Action

A multi-sensor fusion model for autonomous driving that predicts:
1. Ego waypoints for closed-loop control
2. 3D object detection (vehicles, pedestrians, cyclists)

Architecture:
- LiDAR encoder: PointPillars
- Camera encoder: Lift-Splat-Shoot
- BEV backbone: ResNet-style
- Detection head: CenterPoint
- Planning head: Transformer cross-attention
"""

__version__ = "0.1.0"
__author__ = "Fusion2Drive Team"

from wod_fusion.models import FusionModel, FusionModelConfig
from wod_fusion.data import WaymoDataset, WaymoDataModule
from wod_fusion.training import Trainer, TrainerConfig
from wod_fusion.eval import Evaluator
from wod_fusion.export import ModelExporter

__all__ = [
    "FusionModel",
    "FusionModelConfig",
    "WaymoDataset", 
    "WaymoDataModule",
    "Trainer",
    "TrainerConfig",
    "Evaluator",
    "ModelExporter",
]
