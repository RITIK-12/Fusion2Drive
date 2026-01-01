"""
Data module for Waymo Open Dataset handling.

Supports:
- Waymo Open Dataset v2.0.1 (Parquet format, preferred)
- Waymo Open Dataset v1.4.3 (TFRecords format, fallback)
"""

# Lazy imports to avoid issues with missing optional dependencies
def __getattr__(name):
    if name == "WaymoDataset":
        from wod_fusion.data.dataset import WaymoDataset
        return WaymoDataset
    elif name == "WaymoDataModule":
        from wod_fusion.data.datamodule import WaymoDataModule
        return WaymoDataModule
    elif name == "ImageTransform":
        from wod_fusion.data.transforms import ImageTransform
        return ImageTransform
    elif name == "PointCloudTransform":
        from wod_fusion.data.transforms import PointCloudTransform
        return PointCloudTransform
    elif name == "SensorCalibration":
        from wod_fusion.data.transforms import SensorCalibration
        return SensorCalibration
    elif name == "CacheBuilder":
        from wod_fusion.data.cache import CacheBuilder
        return CacheBuilder
    raise AttributeError(f"module 'wod_fusion.data' has no attribute '{name}'")
