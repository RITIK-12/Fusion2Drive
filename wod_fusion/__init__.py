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

# Lazy imports to avoid circular dependencies and missing optional deps
def __getattr__(name):
    if name == "FusionModel":
        from wod_fusion.models import FusionModel
        return FusionModel
    elif name == "FusionModelConfig":
        from wod_fusion.models import FusionModelConfig
        return FusionModelConfig
    elif name == "WaymoDataset":
        from wod_fusion.data.dataset import WaymoDataset
        return WaymoDataset
    elif name == "WaymoDataModule":
        from wod_fusion.data.datamodule import WaymoDataModule
        return WaymoDataModule
    elif name == "Trainer":
        from wod_fusion.training import Trainer
        return Trainer
    elif name == "TrainerConfig":
        from wod_fusion.training import TrainerConfig
        return TrainerConfig
    elif name == "Evaluator":
        from wod_fusion.eval import Evaluator
        return Evaluator
    elif name == "ModelExporter":
        from wod_fusion.export import ModelExporter
        return ModelExporter
    raise AttributeError(f"module 'wod_fusion' has no attribute '{name}'")

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
