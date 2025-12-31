"""
Main fusion model combining LiDAR and camera encoders.

Architecture:
1. PointPillars encoder for LiDAR → BEV features
2. Lift-Splat-Shoot encoder for cameras → BEV features  
3. BEV fusion backbone
4. CenterPoint detection head
5. Waypoint planning head
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from wod_fusion.models.lidar_encoder import PointPillarsEncoder
from wod_fusion.models.camera_encoder import LiftSplatEncoder
from wod_fusion.models.bev_backbone import BEVBackbone
from wod_fusion.models.detection_head import CenterPointHead
from wod_fusion.models.planning_head import WaypointHead

logger = logging.getLogger(__name__)


@dataclass
class FusionModelConfig:
    """Configuration for the fusion model."""
    # BEV grid settings
    bev_x_range: Tuple[float, float] = (-75.0, 75.0)
    bev_y_range: Tuple[float, float] = (-75.0, 75.0)
    bev_resolution: float = 0.5  # meters per pixel
    
    # LiDAR encoder
    lidar_in_channels: int = 4  # x, y, z, intensity
    lidar_voxel_size: Tuple[float, float, float] = (0.2, 0.2, 8.0)
    lidar_max_points_per_voxel: int = 32
    lidar_max_voxels: int = 40000
    lidar_hidden_channels: int = 64
    lidar_out_channels: int = 128
    
    # Camera encoder
    camera_backbone: str = "resnet18"
    camera_image_size: Tuple[int, int] = (640, 480)
    camera_num_cameras: int = 5
    camera_depth_channels: int = 64
    camera_depth_min: float = 1.0
    camera_depth_max: float = 60.0
    camera_out_channels: int = 128
    
    # BEV backbone
    bev_in_channels: int = 256  # lidar + camera
    bev_hidden_channels: int = 256
    bev_out_channels: int = 256
    bev_num_layers: int = 4
    
    # Detection head
    detection_num_classes: int = 3  # vehicle, pedestrian, cyclist
    detection_heatmap_kernel: int = 3
    detection_head_channels: int = 64
    
    # Planning head  
    planning_num_waypoints: int = 10
    planning_hidden_dim: int = 256
    planning_num_layers: int = 3
    
    # Training settings
    use_gradient_checkpointing: bool = False
    
    # Modality settings
    use_lidar: bool = True
    use_camera: bool = True
    
    @property
    def bev_size(self) -> Tuple[int, int]:
        """Compute BEV grid size in pixels."""
        x_size = int((self.bev_x_range[1] - self.bev_x_range[0]) / self.bev_resolution)
        y_size = int((self.bev_y_range[1] - self.bev_y_range[0]) / self.bev_resolution)
        return (x_size, y_size)
    
    @property
    def point_cloud_range(self) -> List[float]:
        """Get point cloud range for LiDAR processing."""
        return [
            self.bev_x_range[0], self.bev_y_range[0], -3.0,
            self.bev_x_range[1], self.bev_y_range[1], 5.0,
        ]


class FusionModel(nn.Module):
    """
    Multi-sensor BEV fusion model for autonomous driving.
    
    Fuses LiDAR and camera inputs in BEV space and outputs:
    1. 3D object detections (vehicles, pedestrians, cyclists)
    2. Future ego waypoints for planning
    
    The architecture is modular and supports:
    - LiDAR-only mode
    - Camera-only mode  
    - Full fusion mode
    
    Designed for:
    - Training on A100 GPUs with mixed precision
    - Inference on Apple Silicon via MPS/CoreML/ONNX
    """
    
    def __init__(self, config: Optional[FusionModelConfig] = None):
        """
        Initialize fusion model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or FusionModelConfig()
        
        # Initialize encoders based on modality settings
        self.lidar_encoder = None
        self.camera_encoder = None
        
        if self.config.use_lidar:
            self.lidar_encoder = PointPillarsEncoder(
                in_channels=self.config.lidar_in_channels,
                voxel_size=list(self.config.lidar_voxel_size),
                point_cloud_range=self.config.point_cloud_range,
                max_points_per_voxel=self.config.lidar_max_points_per_voxel,
                max_voxels=self.config.lidar_max_voxels,
                hidden_channels=self.config.lidar_hidden_channels,
                out_channels=self.config.lidar_out_channels,
            )
        
        if self.config.use_camera:
            self.camera_encoder = LiftSplatEncoder(
                backbone=self.config.camera_backbone,
                image_size=self.config.camera_image_size,
                num_cameras=self.config.camera_num_cameras,
                depth_channels=self.config.camera_depth_channels,
                depth_min=self.config.camera_depth_min,
                depth_max=self.config.camera_depth_max,
                bev_x_range=self.config.bev_x_range,
                bev_y_range=self.config.bev_y_range,
                bev_resolution=self.config.bev_resolution,
                out_channels=self.config.camera_out_channels,
            )
        
        # Compute fusion input channels
        fusion_in_channels = 0
        if self.config.use_lidar:
            fusion_in_channels += self.config.lidar_out_channels
        if self.config.use_camera:
            fusion_in_channels += self.config.camera_out_channels
        
        # Fusion layer (project to common channels)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(fusion_in_channels, self.config.bev_in_channels, 1),
            nn.BatchNorm2d(self.config.bev_in_channels),
            nn.ReLU(inplace=True),
        )
        
        # BEV backbone
        self.bev_backbone = BEVBackbone(
            in_channels=self.config.bev_in_channels,
            hidden_channels=self.config.bev_hidden_channels,
            out_channels=self.config.bev_out_channels,
            num_layers=self.config.bev_num_layers,
        )
        
        # Detection head
        self.detection_head = CenterPointHead(
            in_channels=self.config.bev_out_channels,
            num_classes=self.config.detection_num_classes,
            head_channels=self.config.detection_head_channels,
            heatmap_kernel=self.config.detection_heatmap_kernel,
        )
        
        # Planning head
        self.planning_head = WaypointHead(
            in_channels=self.config.bev_out_channels,
            num_waypoints=self.config.planning_num_waypoints,
            hidden_dim=self.config.planning_hidden_dim,
            num_layers=self.config.planning_num_layers,
            bev_size=self.config.bev_size,
        )
        
        # Gradient checkpointing
        self.use_gradient_checkpointing = self.config.use_gradient_checkpointing
        
        logger.info(f"Initialized FusionModel with config: {self.config}")
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        points_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Camera images [B, N, 3, H, W]
            intrinsics: Camera intrinsics [B, N, 3, 3]
            extrinsics: Camera extrinsics [B, N, 4, 4]
            points: LiDAR points [B, P, 4]
            points_mask: Valid point mask [B, P]
            
        Returns:
            Dict containing:
            - heatmap: Detection heatmap [B, C, H_bev, W_bev]
            - box_reg: Box regression [B, 8, H_bev, W_bev]
            - waypoints: Predicted waypoints [B, K, 3]
            - bev_features: BEV features [B, C, H_bev, W_bev]
        """
        batch_size = images.shape[0] if images is not None else points.shape[0]
        device = images.device if images is not None else points.device
        
        # Encode modalities
        bev_features_list = []
        
        if self.config.use_lidar and points is not None:
            if self.use_gradient_checkpointing and self.training:
                lidar_bev = torch.utils.checkpoint.checkpoint(
                    self.lidar_encoder, points, points_mask,
                    use_reentrant=False,
                )
            else:
                lidar_bev = self.lidar_encoder(points, points_mask)
            bev_features_list.append(lidar_bev)
        
        if self.config.use_camera and images is not None:
            if self.use_gradient_checkpointing and self.training:
                camera_bev = torch.utils.checkpoint.checkpoint(
                    self.camera_encoder, images, intrinsics, extrinsics,
                    use_reentrant=False,
                )
            else:
                camera_bev = self.camera_encoder(images, intrinsics, extrinsics)
            bev_features_list.append(camera_bev)
        
        # Fuse BEV features
        if len(bev_features_list) > 1:
            bev_fused = torch.cat(bev_features_list, dim=1)
        else:
            bev_fused = bev_features_list[0]
        
        bev_fused = self.fusion_proj(bev_fused)
        
        # BEV backbone
        bev_features = self.bev_backbone(bev_fused)
        
        # Detection head
        detection_output = self.detection_head(bev_features)
        
        # Planning head
        waypoints = self.planning_head(bev_features)
        
        return {
            "heatmap": detection_output["heatmap"],
            "box_reg": detection_output["box_reg"],
            "waypoints": waypoints,
            "bev_features": bev_features,
        }
    
    def inference(
        self,
        images: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        points_mask: Optional[torch.Tensor] = None,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference with post-processing.
        
        Args:
            images: Camera images [B, N, 3, H, W]
            intrinsics: Camera intrinsics [B, N, 3, 3]
            extrinsics: Camera extrinsics [B, N, 4, 4]
            points: LiDAR points [B, P, 4]
            points_mask: Valid point mask [B, P]
            score_threshold: Detection score threshold
            nms_threshold: NMS IoU threshold
            
        Returns:
            Dict containing:
            - detections: List of detection dicts per sample
            - waypoints: Predicted waypoints [B, K, 3]
        """
        # Forward pass
        outputs = self.forward(
            images=images,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            points=points,
            points_mask=points_mask,
        )
        
        # Post-process detections
        detections = self.detection_head.decode(
            outputs["heatmap"],
            outputs["box_reg"],
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            bev_x_range=self.config.bev_x_range,
            bev_y_range=self.config.bev_y_range,
            bev_resolution=self.config.bev_resolution,
        )
        
        return {
            "detections": detections,
            "waypoints": outputs["waypoints"],
        }
    
    @classmethod
    def from_config(cls, config_dict: Dict) -> "FusionModel":
        """Create model from config dictionary."""
        config = FusionModelConfig(**config_dict)
        return cls(config)
    
    @classmethod
    def load_from_checkpoint(
        cls, 
        checkpoint_path: str,
        map_location: str = "cpu",
    ) -> "FusionModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        config_dict = checkpoint.get("config", {})
        model = cls.from_config(config_dict)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return model
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts by component."""
        counts = {}
        
        if self.lidar_encoder is not None:
            counts["lidar_encoder"] = sum(p.numel() for p in self.lidar_encoder.parameters())
        
        if self.camera_encoder is not None:
            counts["camera_encoder"] = sum(p.numel() for p in self.camera_encoder.parameters())
        
        counts["fusion_proj"] = sum(p.numel() for p in self.fusion_proj.parameters())
        counts["bev_backbone"] = sum(p.numel() for p in self.bev_backbone.parameters())
        counts["detection_head"] = sum(p.numel() for p in self.detection_head.parameters())
        counts["planning_head"] = sum(p.numel() for p in self.planning_head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        
        return counts


class FusionModelLite(FusionModel):
    """
    Lightweight version of FusionModel for Mac deployment.
    
    Differences from full model:
    - Smaller backbone (fewer channels)
    - Reduced BEV resolution
    - Fewer cameras
    """
    
    @classmethod
    def get_lite_config(cls) -> FusionModelConfig:
        """Get configuration for lite model."""
        return FusionModelConfig(
            # Reduced BEV grid
            bev_x_range=(-50.0, 50.0),
            bev_y_range=(-50.0, 50.0),
            bev_resolution=0.5,
            
            # Smaller LiDAR encoder
            lidar_hidden_channels=32,
            lidar_out_channels=64,
            lidar_max_voxels=20000,
            
            # Smaller camera encoder
            camera_backbone="resnet18",
            camera_image_size=(480, 320),
            camera_num_cameras=3,  # FRONT, FRONT_LEFT, FRONT_RIGHT
            camera_depth_channels=32,
            camera_out_channels=64,
            
            # Smaller BEV backbone
            bev_in_channels=128,
            bev_hidden_channels=128,
            bev_out_channels=128,
            bev_num_layers=3,
            
            # Smaller heads
            detection_head_channels=32,
            planning_hidden_dim=128,
            planning_num_layers=2,
        )
    
    def __init__(self, config: Optional[FusionModelConfig] = None):
        if config is None:
            config = self.get_lite_config()
        super().__init__(config)
