"""
Lift-Splat-Shoot style camera encoder.

Converts multi-camera images to BEV features via:
1. Image backbone (EfficientNet/I-JEPA/ConvNeXt) for per-camera features
2. Depth distribution prediction per pixel
3. Lift: Create 3D frustum features by outer product of image features and depth
4. Splat: Accumulate frustum features into BEV grid
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

logger = logging.getLogger(__name__)


# ============================================================================
# I-JEPA Implementation (Self-supervised Vision Transformer)
# ============================================================================

class PatchEmbed(nn.Module):
    """Image to Patch Embedding for I-JEPA."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class Attention(nn.Module):
    """Multi-head self-attention for I-JEPA."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block for I-JEPA."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class IJEPABlock(nn.Module):
    """Transformer block for I-JEPA."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class IJEPAEncoder(nn.Module):
    """
    I-JEPA (Image-based Joint-Embedding Predictive Architecture) encoder.
    
    Based on: https://arxiv.org/abs/2301.08243
    Uses Vision Transformer architecture with self-supervised pretraining objective.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 512,
        drop_rate: float = 0.0,
    ):
        """
        Initialize I-JEPA encoder.
        
        Args:
            img_size: Input image size (assumes square)
            patch_size: Patch size for tokenization
            in_channels: Input channels (3 for RGB)
            embed_dim: Transformer embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            out_channels: Output feature channels
            drop_rate: Dropout rate
        """
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            IJEPABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to output channels and reshape to spatial
        self.head = nn.Linear(embed_dim, out_channels)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        
        # Store feature channels for FPN compatibility
        self.feature_channels = [out_channels, out_channels, out_channels]
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate position embeddings for different input sizes."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N and h == w:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        dim = x.shape[-1]
        h0 = h // self.patch_size
        w0 = w // self.patch_size
        
        # Interpolate
        sqrt_N = int(math.sqrt(N))
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat([class_pos_embed.unsqueeze(0), patch_pos_embed], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using I-JEPA encoder.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            [B, C, H/8, W/8] feature map
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding (interpolate if needed)
        pos_embed = self.interpolate_pos_embed(x, H, W)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Remove cls token and project
        x = x[:, 1:]  # [B, N, D]
        x = self.head(x)  # [B, N, out_channels]
        
        # Reshape to spatial format
        h = H // self.patch_size
        w = W // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.out_channels, h, w)
        
        # Upsample to 1/8 resolution (patch_size=16 gives 1/16, we want 1/8)
        target_h = H // 8
        target_w = W // 8
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return x


# ============================================================================
# Image Backbone (EfficientNet / I-JEPA / timm models)
# ============================================================================

class ImageBackbone(nn.Module):
    """
    Image feature extractor using pretrained backbones.
    
    Supports: efficientnet_b0-b7, ijepa_small/base/large, convnext_tiny, resnet18/34/50
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        out_channels: int = 512,
    ):
        """
        Initialize image backbone.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            out_channels: Output feature channels
        """
        super().__init__()
        
        self.backbone_name = backbone
        
        if backbone.startswith("ijepa"):
            # I-JEPA variants
            configs = {
                "ijepa_small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
                "ijepa_base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
                "ijepa_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
            }
            cfg = configs.get(backbone, configs["ijepa_base"])
            
            self.encoder = IJEPAEncoder(
                embed_dim=cfg["embed_dim"],
                depth=cfg["depth"],
                num_heads=cfg["num_heads"],
                out_channels=out_channels,
            )
            self.feature_channels = [out_channels, out_channels, out_channels]
            self.use_ijepa = True
            self.neck = None
            
        elif HAS_TIMM and backbone.startswith("efficientnet"):
            # EfficientNet via timm
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[2, 3, 4],  # Get multi-scale features
            )
            self.feature_channels = self.encoder.feature_info.channels()
            self.use_ijepa = False
            
            # FPN-style neck to combine multi-scale features
            total_channels = sum(self.feature_channels)
            self.neck = nn.Sequential(
                nn.Conv2d(total_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
        elif HAS_TIMM:
            # Other timm backbones (convnext, resnet, etc.)
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=[2, 3, 4],
            )
            self.feature_channels = self.encoder.feature_info.channels()
            self.use_ijepa = False
            
            total_channels = sum(self.feature_channels)
            self.neck = nn.Sequential(
                nn.Conv2d(total_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
        else:
            # Fallback to torchvision EfficientNet
            import torchvision.models as models
            
            if hasattr(models, 'efficientnet_b0'):
                effnet = models.efficientnet_b0(pretrained=pretrained)
                self.feature_channels = [40, 112, 320]  # EfficientNet-B0 channels
            else:
                # Ultimate fallback to ResNet
                logger.warning("EfficientNet not available, falling back to ResNet18")
                resnet = models.resnet18(pretrained=pretrained)
                self.feature_channels = [128, 256, 512]
                
                self.layer0 = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
                )
                self.layer1 = resnet.layer1
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4
                self.encoder = None
            
            self.use_ijepa = False
            total_channels = sum(self.feature_channels)
            self.neck = nn.Sequential(
                nn.Conv2d(total_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.
        
        Args:
            x: [B, 3, H, W] input image
            
        Returns:
            [B, C, H/8, W/8] feature map
        """
        if self.use_ijepa:
            # I-JEPA returns features directly
            return self.encoder(x)
        
        if self.encoder is not None:
            # timm backbone
            features = self.encoder(x)
        else:
            # torchvision backbone fallback
            x = self.layer0(x)
            x = self.layer1(x)
            f1 = self.layer2(x)   # 1/8
            f2 = self.layer3(f1)  # 1/16
            f3 = self.layer4(f2)  # 1/32
            features = [f1, f2, f3]
        
        # Upsample all features to same resolution (1/8)
        target_size = features[0].shape[2:]
        upsampled = [features[0]]
        for f in features[1:]:
            upsampled.append(
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            )
        
        # Concatenate and project
        combined = torch.cat(upsampled, dim=1)
        out = self.neck(combined)
        
        return out


class DepthNet(nn.Module):
    """
    Predict depth distribution for each pixel.
    
    Outputs a discrete depth distribution over D bins,
    representing the probability of each depth value.
    """
    
    def __init__(
        self,
        in_channels: int,
        depth_channels: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
    ):
        """
        Initialize depth network.
        
        Args:
            in_channels: Input feature channels
            depth_channels: Number of depth bins
            depth_min: Minimum depth in meters
            depth_max: Maximum depth in meters
        """
        super().__init__()
        
        self.depth_channels = depth_channels
        self.depth_min = depth_min
        self.depth_max = depth_max
        
        # Depth distribution predictor
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, depth_channels, 1),
        )
        
        # Register depth bins
        depth_bins = torch.linspace(depth_min, depth_max, depth_channels)
        self.register_buffer("depth_bins", depth_bins)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict depth distribution.
        
        Args:
            x: [B, C, H, W] image features
            
        Returns:
            Tuple of:
            - depth_probs: [B, D, H, W] depth distribution
            - depth_values: [D] depth bin values
        """
        logits = self.net(x)
        depth_probs = F.softmax(logits, dim=1)
        
        return depth_probs, self.depth_bins


class LiftSplatEncoder(nn.Module):
    """
    Lift-Splat-Shoot style multi-camera to BEV encoder.
    
    For each camera:
    1. Extract image features with CNN/ViT backbone
    2. Predict depth distribution per pixel
    3. Lift features to 3D frustum
    4. Transform frustum to ego frame using extrinsics
    5. Splat (accumulate) features into BEV grid
    
    Combines all cameras into a single BEV feature map.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        image_size: Tuple[int, int] = (640, 480),
        num_cameras: int = 5,
        depth_channels: int = 64,
        depth_min: float = 1.0,
        depth_max: float = 60.0,
        bev_x_range: Tuple[float, float] = (-75.0, 75.0),
        bev_y_range: Tuple[float, float] = (-75.0, 75.0),
        bev_resolution: float = 0.5,
        out_channels: int = 128,
    ):
        """
        Initialize Lift-Splat encoder.
        
        Args:
            backbone: Image backbone architecture (efficientnet_b0, ijepa_base, etc.)
            image_size: (width, height) of input images
            num_cameras: Number of camera views
            depth_channels: Number of depth bins
            depth_min: Minimum depth in meters
            depth_max: Maximum depth in meters
            bev_x_range: (min, max) x range of BEV grid in meters
            bev_y_range: (min, max) y range of BEV grid in meters
            bev_resolution: BEV grid resolution in meters per pixel
            out_channels: Output BEV feature channels
        """
        super().__init__()
        
        self.image_size = image_size
        self.num_cameras = num_cameras
        self.depth_channels = depth_channels
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_resolution = bev_resolution
        
        # Compute BEV grid size
        self.bev_h = int((bev_y_range[1] - bev_y_range[0]) / bev_resolution)
        self.bev_w = int((bev_x_range[1] - bev_x_range[0]) / bev_resolution)
        
        # Image backbone (shared across cameras)
        self.backbone = ImageBackbone(
            backbone=backbone,
            pretrained=True,
            out_channels=out_channels,
        )
        
        # Depth predictor (shared across cameras)
        self.depth_net = DepthNet(
            in_channels=self.backbone.out_channels,
            depth_channels=depth_channels,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        
        # BEV encoder (after splatting)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels
        
        # Create frustum grid (pre-computed for efficiency)
        self._create_frustum()
    
    def _create_frustum(self):
        """Create frustum grid for lifting."""
        # Feature map size (1/8 of image)
        feat_h = self.image_size[1] // 8
        feat_w = self.image_size[0] // 8
        
        # Create grid of (u, v, d) coordinates
        us = torch.linspace(0, self.image_size[0] - 1, feat_w)
        vs = torch.linspace(0, self.image_size[1] - 1, feat_h)
        ds = torch.linspace(
            self.depth_net.depth_min,
            self.depth_net.depth_max,
            self.depth_channels,
        )
        
        # Grid: [D, H, W, 3] -> (u, v, d)
        D, H, W = len(ds), len(vs), len(us)
        
        us = us.view(1, 1, W).expand(D, H, W)
        vs = vs.view(1, H, 1).expand(D, H, W)
        ds = ds.view(D, 1, 1).expand(D, H, W)
        
        frustum = torch.stack([us, vs, ds], dim=-1)  # [D, H, W, 3]
        
        self.register_buffer("frustum", frustum)
        self.feat_h = feat_h
        self.feat_w = feat_w
    
    def _get_geometry(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 3D points from frustum using camera geometry.
        
        Args:
            intrinsics: [B, N, 3, 3] camera intrinsics
            extrinsics: [B, N, 4, 4] camera-to-ego transforms
            
        Returns:
            [B, N, D, H, W, 3] 3D points in ego frame
        """
        B, N = intrinsics.shape[:2]
        D, H, W = self.depth_channels, self.feat_h, self.feat_w
        device = intrinsics.device
        
        # Get frustum points [D, H, W, 3]
        frustum = self.frustum.to(device)
        
        # Unproject to camera frame
        # (u, v, d) -> (x, y, z) in camera frame
        points = []
        
        for b in range(B):
            batch_points = []
            for n in range(N):
                K = intrinsics[b, n]  # [3, 3]
                E = extrinsics[b, n]  # [4, 4]
                
                # Extract intrinsic parameters
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                
                # Scale to feature map resolution
                fx_feat = fx / 8
                fy_feat = fy / 8
                cx_feat = cx / 8
                cy_feat = cy / 8
                
                # Unproject: (u, v, d) -> (x, y, z) in camera frame
                u = frustum[..., 0]  # [D, H, W]
                v = frustum[..., 1]
                d = frustum[..., 2]
                
                # Scale u, v to feature map coords
                u_feat = u / 8
                v_feat = v / 8
                
                x_cam = (u_feat - cx_feat) * d / fx_feat
                y_cam = (v_feat - cy_feat) * d / fy_feat
                z_cam = d
                
                # Stack as homogeneous coordinates
                pts_cam = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1)
                
                # Transform to ego frame
                pts_cam = pts_cam.view(-1, 4)  # [D*H*W, 4]
                pts_ego = (E @ pts_cam.T).T  # [D*H*W, 4]
                pts_ego = pts_ego[:, :3].view(D, H, W, 3)  # [D, H, W, 3]
                
                batch_points.append(pts_ego)
            
            points.append(torch.stack(batch_points, dim=0))
        
        return torch.stack(points, dim=0)  # [B, N, D, H, W, 3]
    
    def _splat_to_bev(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
        depth_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Splat 3D features to BEV grid.
        
        Args:
            features: [B, N, C, H, W] image features
            points: [B, N, D, H, W, 3] 3D points in ego frame
            depth_probs: [B, N, D, H, W] depth distributions
            
        Returns:
            [B, C, bev_H, bev_W] BEV feature map
        """
        B, N, C, H, W = features.shape
        D = self.depth_channels
        device = features.device
        
        # Initialize BEV grid
        bev = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
        bev_count = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)
        
        # Weighted features by depth probability
        # features: [B, N, C, H, W] -> [B, N, C, D, H, W]
        features_expanded = features[:, :, :, None, :, :].expand(-1, -1, -1, D, -1, -1)
        depth_probs_expanded = depth_probs[:, :, None, :, :, :]  # [B, N, 1, D, H, W]
        
        weighted_features = features_expanded * depth_probs_expanded  # [B, N, C, D, H, W]
        
        # Splat each point to BEV
        for b in range(B):
            for n in range(N):
                pts = points[b, n].view(-1, 3)  # [D*H*W, 3]
                feats = weighted_features[b, n].permute(1, 2, 3, 0).reshape(-1, C)  # [D*H*W, C]
                
                # Compute BEV indices
                x_idx = ((pts[:, 0] - self.bev_x_range[0]) / self.bev_resolution).long()
                y_idx = ((pts[:, 1] - self.bev_y_range[0]) / self.bev_resolution).long()
                
                # Valid mask
                valid = (
                    (x_idx >= 0) & (x_idx < self.bev_w) &
                    (y_idx >= 0) & (y_idx < self.bev_h) &
                    (pts[:, 2] > -3) & (pts[:, 2] < 5)  # Height filter
                )
                
                x_idx = x_idx[valid]
                y_idx = y_idx[valid]
                feats = feats[valid]
                
                # Accumulate features (simplified - use scatter_add for efficiency)
                if x_idx.numel() > 0:
                    bev_idx = y_idx * self.bev_w + x_idx
                    bev_flat = bev[b].view(C, -1)
                    count_flat = bev_count[b].view(1, -1)
                    
                    # Scatter add
                    bev_flat.scatter_add_(1, bev_idx.unsqueeze(0).expand(C, -1), feats.T)
                    count_flat.scatter_add_(1, bev_idx.unsqueeze(0), torch.ones_like(bev_idx.unsqueeze(0).float()))
        
        # Average pooling (divide by count)
        bev = bev / (bev_count + 1e-6)
        
        return bev
    
    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode multi-camera images to BEV features.
        
        Args:
            images: [B, N, 3, H, W] multi-camera images
            intrinsics: [B, N, 3, 3] camera intrinsics
            extrinsics: [B, N, 4, 4] camera-to-ego transforms
            
        Returns:
            [B, C, bev_H, bev_W] BEV feature map
        """
        B, N, _, H, W = images.shape
        
        # Flatten batch and cameras
        images_flat = images.view(B * N, 3, H, W)
        
        # Extract image features
        features = self.backbone(images_flat)  # [B*N, C, H/8, W/8]
        
        # Predict depth distributions
        depth_probs, _ = self.depth_net(features)  # [B*N, D, H/8, W/8]
        
        # Reshape back to batch and cameras
        C = features.shape[1]
        feat_h, feat_w = features.shape[2:]
        features = features.view(B, N, C, feat_h, feat_w)
        depth_probs = depth_probs.view(B, N, self.depth_channels, feat_h, feat_w)
        
        # Compute 3D geometry
        points = self._get_geometry(intrinsics, extrinsics)  # [B, N, D, H, W, 3]
        
        # Splat to BEV
        bev = self._splat_to_bev(features, points, depth_probs)
        
        # Apply BEV encoder
        bev = self.bev_encoder(bev)
        
        return bev
