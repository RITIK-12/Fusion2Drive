"""
PointPillars-style LiDAR encoder.

Converts raw point cloud to BEV features via:
1. Voxelization into pillars (vertical columns)
2. PointNet-style feature extraction per pillar
3. Scatter to 2D pseudo-image (BEV grid)
4. 2D CNN for spatial feature learning
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class Voxelizer(nn.Module):
    """
    Voxelize point cloud into pillars.
    
    Each pillar is a vertical column in the BEV grid.
    Points within each pillar are processed together.
    """
    
    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_points_per_voxel: int = 32,
        max_voxels: int = 40000,
    ):
        """
        Initialize voxelizer.
        
        Args:
            voxel_size: [x, y, z] voxel size in meters
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points_per_voxel: Maximum points per voxel
            max_voxels: Maximum number of voxels
        """
        super().__init__()
        
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        self.max_points_per_voxel = max_points_per_voxel
        self.max_voxels = max_voxels
        
        # Compute grid size
        grid_size = (
            self.point_cloud_range[3:] - self.point_cloud_range[:3]
        ) / self.voxel_size
        self.grid_size = grid_size.long().tolist()
    
    def forward(
        self,
        points: torch.Tensor,
        points_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Voxelize point cloud.
        
        Args:
            points: [B, N, 4] points (x, y, z, intensity)
            points_mask: [B, N] valid point mask
            
        Returns:
            Tuple of:
            - voxels: [B, V, P, C] voxel features
            - coords: [B, V, 3] voxel coordinates (z, y, x)
            - num_points: [B, V] number of points per voxel
        """
        batch_size = points.shape[0]
        device = points.device
        
        # Move grid params to device
        voxel_size = self.voxel_size.to(device)
        pc_range = self.point_cloud_range.to(device)
        
        all_voxels = []
        all_coords = []
        all_num_points = []
        
        for b in range(batch_size):
            pts = points[b]
            
            # Apply mask if provided
            if points_mask is not None:
                pts = pts[points_mask[b]]
            
            # Filter points outside range
            mask = (
                (pts[:, 0] >= pc_range[0]) & (pts[:, 0] < pc_range[3]) &
                (pts[:, 1] >= pc_range[1]) & (pts[:, 1] < pc_range[4]) &
                (pts[:, 2] >= pc_range[2]) & (pts[:, 2] < pc_range[5])
            )
            pts = pts[mask]
            
            if pts.shape[0] == 0:
                # No valid points - return empty voxels
                voxels = torch.zeros(
                    self.max_voxels, self.max_points_per_voxel, 4,
                    device=device, dtype=torch.float32
                )
                coords = torch.zeros(self.max_voxels, 3, device=device, dtype=torch.long)
                num_points = torch.zeros(self.max_voxels, device=device, dtype=torch.long)
            else:
                # Compute voxel indices
                voxel_idx = ((pts[:, :3] - pc_range[:3]) / voxel_size).long()
                
                # Clamp to grid bounds
                voxel_idx[:, 0] = voxel_idx[:, 0].clamp(0, self.grid_size[0] - 1)
                voxel_idx[:, 1] = voxel_idx[:, 1].clamp(0, self.grid_size[1] - 1)
                voxel_idx[:, 2] = voxel_idx[:, 2].clamp(0, self.grid_size[2] - 1)
                
                # Hash voxels for unique identification
                voxel_hash = (
                    voxel_idx[:, 0] * self.grid_size[1] * self.grid_size[2] +
                    voxel_idx[:, 1] * self.grid_size[2] +
                    voxel_idx[:, 2]
                )
                
                # Get unique voxels
                unique_hash, inverse = torch.unique(voxel_hash, return_inverse=True)
                num_voxels = min(unique_hash.shape[0], self.max_voxels)
                
                # Initialize outputs
                voxels = torch.zeros(
                    self.max_voxels, self.max_points_per_voxel, 4,
                    device=device, dtype=torch.float32
                )
                coords = torch.zeros(self.max_voxels, 3, device=device, dtype=torch.long)
                num_points = torch.zeros(self.max_voxels, device=device, dtype=torch.long)
                
                # Fill voxels (simplified - for production use scatter operations)
                for v_idx in range(num_voxels):
                    point_mask = inverse == v_idx
                    point_indices = torch.where(point_mask)[0]
                    
                    n_pts = min(point_indices.shape[0], self.max_points_per_voxel)
                    voxels[v_idx, :n_pts] = pts[point_indices[:n_pts]]
                    coords[v_idx] = voxel_idx[point_indices[0]]
                    num_points[v_idx] = n_pts
            
            all_voxels.append(voxels)
            all_coords.append(coords)
            all_num_points.append(num_points)
        
        return (
            torch.stack(all_voxels),
            torch.stack(all_coords),
            torch.stack(all_num_points),
        )


class PillarFeatureNet(nn.Module):
    """
    PointNet-style feature extraction for each pillar.
    
    For each point, augment features with:
    - Offset from pillar center (x, y, z)
    - Offset from point cloud center
    
    Then apply shared MLPs and max pooling.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        out_channels: int = 64,
    ):
        """
        Initialize pillar feature network.
        
        Args:
            in_channels: Input point features (x, y, z, intensity)
            hidden_channels: Hidden layer channels
            out_channels: Output feature channels
        """
        super().__init__()
        
        # Augmented features: original + pillar center offset (3) + cluster offset (3)
        augmented_channels = in_channels + 3 + 3
        
        self.net = nn.Sequential(
            nn.Linear(augmented_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def forward(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from voxels.
        
        Args:
            voxels: [B, V, P, C] voxel features
            num_points: [B, V] number of points per voxel
            
        Returns:
            [B, V, out_channels] pillar features
        """
        batch_size, num_voxels, max_points, in_channels = voxels.shape
        device = voxels.device
        
        # Compute pillar centers
        # Mean of points in each pillar (masked by num_points)
        mask = torch.arange(max_points, device=device)[None, None, :] < num_points[:, :, None]
        mask = mask.unsqueeze(-1).float()
        
        pillar_sum = (voxels * mask).sum(dim=2)
        pillar_count = mask.sum(dim=2).clamp(min=1)
        pillar_center = pillar_sum / pillar_count
        
        # Compute offsets from pillar center
        center_offset = voxels[:, :, :, :3] - pillar_center[:, :, None, :3]
        
        # Compute offsets from cluster center (mean of all valid points)
        # Simplified: use pillar center as cluster center
        cluster_offset = center_offset.clone()
        
        # Augment features
        augmented = torch.cat([voxels, center_offset, cluster_offset], dim=-1)
        
        # Flatten for MLP
        augmented = augmented.view(-1, augmented.shape[-1])
        
        # Apply MLP
        features = self.net(augmented)
        
        # Reshape back
        features = features.view(batch_size, num_voxels, max_points, -1)
        
        # Max pooling over points
        features = features * mask
        features = features.max(dim=2)[0]  # [B, V, C]
        
        return features


class PillarScatter(nn.Module):
    """
    Scatter pillar features to 2D BEV pseudo-image.
    """
    
    def __init__(
        self,
        in_channels: int,
        grid_size: List[int],
    ):
        """
        Initialize scatter module.
        
        Args:
            in_channels: Pillar feature channels
            grid_size: [X, Y, Z] grid dimensions
        """
        super().__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size  # [X, Y, Z]
    
    def forward(
        self,
        pillar_features: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scatter pillar features to BEV grid.
        
        Args:
            pillar_features: [B, V, C] pillar features
            coords: [B, V, 3] voxel coordinates (x, y, z)
            
        Returns:
            [B, C, Y, X] BEV feature map
        """
        batch_size, num_voxels, channels = pillar_features.shape
        device = pillar_features.device
        
        # Initialize BEV feature map
        bev = torch.zeros(
            batch_size, channels, self.grid_size[1], self.grid_size[0],
            device=device, dtype=pillar_features.dtype,
        )
        
        # Scatter features
        for b in range(batch_size):
            for v in range(num_voxels):
                x, y, z = coords[b, v].long()
                if x >= 0 and x < self.grid_size[0] and y >= 0 and y < self.grid_size[1]:
                    bev[b, :, y, x] = pillar_features[b, v]
        
        return bev


class PointPillarsEncoder(nn.Module):
    """
    Complete PointPillars encoder for LiDAR-to-BEV transformation.
    
    Pipeline:
    1. Voxelization: Point cloud → Pillars
    2. PillarFeatureNet: Per-pillar feature extraction
    3. Scatter: Pillars → 2D pseudo-image
    4. Backbone: 2D CNN for spatial features
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        voxel_size: List[float] = None,
        point_cloud_range: List[float] = None,
        max_points_per_voxel: int = 32,
        max_voxels: int = 40000,
        hidden_channels: int = 64,
        out_channels: int = 128,
    ):
        """
        Initialize PointPillars encoder.
        
        Args:
            in_channels: Input point features
            voxel_size: [x, y, z] voxel size in meters
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points_per_voxel: Maximum points per voxel
            max_voxels: Maximum number of voxels
            hidden_channels: Hidden feature channels
            out_channels: Output feature channels
        """
        super().__init__()
        
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 8.0]
        if point_cloud_range is None:
            point_cloud_range = [-75.0, -75.0, -3.0, 75.0, 75.0, 5.0]
        
        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )
        
        self.pillar_net = PillarFeatureNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        
        self.scatter = PillarScatter(
            in_channels=hidden_channels,
            grid_size=self.voxelizer.grid_size,
        )
        
        # 2D backbone for spatial feature learning
        self.backbone = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def forward(
        self,
        points: torch.Tensor,
        points_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode point cloud to BEV features.
        
        Args:
            points: [B, N, 4] points (x, y, z, intensity)
            points_mask: [B, N] valid point mask
            
        Returns:
            [B, C, H, W] BEV feature map
        """
        # Voxelize
        voxels, coords, num_points = self.voxelizer(points, points_mask)
        
        # Extract pillar features
        pillar_features = self.pillar_net(voxels, num_points)
        
        # Scatter to BEV
        bev = self.scatter(pillar_features, coords)
        
        # Apply backbone
        bev = self.backbone(bev)
        
        return bev
