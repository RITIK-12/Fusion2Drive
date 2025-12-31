"""
BEV backbone for spatial feature learning.

EfficientNet-style 2D CNN with MBConv blocks operating on BEV features.
Uses inverted residuals with squeeze-excitation for efficient feature learning.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# EfficientNet-style Building Blocks
# ============================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation module.
    
    Adaptively recalibrates channel-wise feature responses.
    """
    
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = nn.SiLU(inplace=True)
        self.scale_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.
    
    Core building block of EfficientNet:
    1. Expansion phase (1x1 conv to expand channels)
    2. Depthwise separable convolution (3x3 or 5x5)
    3. Squeeze-and-Excitation
    4. Projection phase (1x1 conv to reduce channels)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ):
        """
        Initialize MBConv block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            expand_ratio: Channel expansion ratio
            kernel_size: Depthwise conv kernel size (3 or 5)
            stride: Stride for downsampling
            se_ratio: Squeeze-excitation reduction ratio
            drop_path_rate: Stochastic depth drop rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion
        expanded_channels = int(in_channels * expand_ratio)
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.expand_conv = nn.Identity()
            expanded_channels = in_channels
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride=stride, padding=padding, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        )
        
        # Squeeze-and-Excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(expanded_channels, squeeze_channels)
        
        # Projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # Stochastic depth (drop path)
        self.drop_path_rate = drop_path_rate
    
    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth during training."""
        if not self.training or self.drop_path_rate == 0:
            return x
        
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        out = self.expand_conv(x)
        
        # Depthwise
        out = self.depthwise_conv(out)
        
        # Squeeze-and-Excitation
        out = self.se(out)
        
        # Projection
        out = self.project_conv(out)
        
        # Residual connection with optional stochastic depth
        if self.use_residual:
            out = self._drop_path(out) + identity
        
        return out


class FusedMBConvBlock(nn.Module):
    """
    Fused MBConv block (used in EfficientNetV2).
    
    Replaces depthwise + 1x1 with a single 3x3 conv for efficiency
    at early stages where input resolution is high.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4.0,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        expanded_channels = int(in_channels * expand_ratio)
        padding = (kernel_size - 1) // 2
        
        # Fused expansion + depthwise
        self.fused_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, expanded_channels, kernel_size,
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
        )
        
        # Optional SE
        if se_ratio > 0:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(expanded_channels, squeeze_channels)
        else:
            self.se = nn.Identity()
        
        # Projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.drop_path_rate = drop_path_rate
    
    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0:
            return x
        
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.fused_conv(x)
        out = self.se(out)
        out = self.project_conv(out)
        
        if self.use_residual:
            out = self._drop_path(out) + identity
        
        return out


# ============================================================================
# Legacy BasicBlock (kept for backwards compatibility)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for BEV backbone (legacy)."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================================
# BEV Backbone (EfficientNet-style)
# ============================================================================

class BEVBackbone(nn.Module):
    """
    BEV feature backbone using EfficientNet-style MBConv blocks.
    
    A multi-scale U-Net style backbone that:
    1. Downsamples BEV features for larger receptive field using MBConv
    2. Upsamples and combines multi-scale features
    3. Outputs same resolution as input
    
    Uses squeeze-excitation and inverted residuals for efficiency.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 4,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.1,
        use_fused_mbconv: bool = True,
    ):
        """
        Initialize BEV backbone.
        
        Args:
            in_channels: Input BEV feature channels
            hidden_channels: Hidden layer channels
            out_channels: Output feature channels
            num_layers: Number of MBConv blocks per scale
            expand_ratio: Channel expansion ratio in MBConv
            se_ratio: Squeeze-excitation reduction ratio
            drop_path_rate: Stochastic depth drop rate
            use_fused_mbconv: Use FusedMBConv at first stage
        """
        super().__init__()
        
        # Initial projection (stem)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        # Calculate drop path rates for each block
        num_blocks = num_layers * 3  # 3 encoder stages
        drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        # Encoder stages with MBConv blocks
        block_idx = 0
        
        # Stage 1: Same resolution (use FusedMBConv for efficiency)
        if use_fused_mbconv:
            self.encoder1 = self._make_fused_stage(
                hidden_channels, hidden_channels, num_layers,
                stride=1, expand_ratio=expand_ratio,
                drop_rates=drop_rates[block_idx:block_idx + num_layers],
            )
        else:
            self.encoder1 = self._make_mbconv_stage(
                hidden_channels, hidden_channels, num_layers,
                stride=1, expand_ratio=expand_ratio, se_ratio=se_ratio,
                drop_rates=drop_rates[block_idx:block_idx + num_layers],
            )
        block_idx += num_layers
        
        # Stage 2: 1/2 resolution
        self.encoder2 = self._make_mbconv_stage(
            hidden_channels, hidden_channels * 2, num_layers,
            stride=2, expand_ratio=expand_ratio, se_ratio=se_ratio,
            drop_rates=drop_rates[block_idx:block_idx + num_layers],
        )
        block_idx += num_layers
        
        # Stage 3: 1/4 resolution
        self.encoder3 = self._make_mbconv_stage(
            hidden_channels * 2, hidden_channels * 2, num_layers,
            stride=2, expand_ratio=expand_ratio, se_ratio=se_ratio,
            drop_rates=drop_rates[block_idx:block_idx + num_layers],
        )
        
        # Decoder (upsampling with skip connections)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.SiLU(inplace=True),
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 2, stride=2),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        
        # Final projection
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        
        self.out_channels = out_channels
    
    def _make_mbconv_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.25,
        drop_rates: Optional[List[float]] = None,
    ) -> nn.Sequential:
        """Create a stage of MBConv blocks."""
        if drop_rates is None:
            drop_rates = [0.0] * num_blocks
        
        layers = []
        
        # First block handles stride and channel change
        layers.append(MBConvBlock(
            in_channels, out_channels,
            expand_ratio=expand_ratio,
            kernel_size=3,
            stride=stride,
            se_ratio=se_ratio,
            drop_path_rate=drop_rates[0],
        ))
        
        # Remaining blocks
        for i in range(1, num_blocks):
            layers.append(MBConvBlock(
                out_channels, out_channels,
                expand_ratio=expand_ratio,
                kernel_size=3,
                stride=1,
                se_ratio=se_ratio,
                drop_path_rate=drop_rates[i],
            ))
        
        return nn.Sequential(*layers)
    
    def _make_fused_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
        drop_rates: Optional[List[float]] = None,
    ) -> nn.Sequential:
        """Create a stage of FusedMBConv blocks."""
        if drop_rates is None:
            drop_rates = [0.0] * num_blocks
        
        layers = []
        
        layers.append(FusedMBConvBlock(
            in_channels, out_channels,
            expand_ratio=expand_ratio,
            kernel_size=3,
            stride=stride,
            drop_path_rate=drop_rates[0],
        ))
        
        for i in range(1, num_blocks):
            layers.append(FusedMBConvBlock(
                out_channels, out_channels,
                expand_ratio=expand_ratio,
                kernel_size=3,
                stride=1,
                drop_path_rate=drop_rates[i],
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, H, W] BEV features
            
        Returns:
            [B, C_out, H, W] processed BEV features
        """
        # Stem
        x = self.stem(x)
        
        # Encoder
        e1 = self.encoder1(x)      # Same resolution
        e2 = self.encoder2(e1)     # 1/2 resolution
        e3 = self.encoder3(e2)     # 1/4 resolution
        
        # Decoder with skip connections
        d3 = self.decoder3(e3)
        
        d2_up = self.upsample2(d3)
        # Handle size mismatch due to odd dimensions
        if d2_up.shape[2:] != e2.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2_up, e2], dim=1))
        
        d1_up = self.upsample1(d2)
        if d1_up.shape[2:] != e1.shape[2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([d1_up, e1], dim=1))
        
        # Final projection
        out = self.final(d1)
        
        return out


class SimpleBEVBackbone(nn.Module):
    """
    Simplified BEV backbone using MBConv blocks without upsampling.
    
    For faster inference and reduced memory usage.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 4,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        ]
        
        # MBConv blocks
        for i in range(num_layers - 1):
            layers.append(MBConvBlock(
                hidden_channels, hidden_channels,
                expand_ratio=expand_ratio,
                kernel_size=3,
                stride=1,
                se_ratio=se_ratio,
            ))
        
        # Final projection
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.net = nn.Sequential(*layers)
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
