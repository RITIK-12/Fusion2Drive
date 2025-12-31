"""
Waypoint prediction head for ego planning.

Predicts future ego waypoints from BEV features.
Uses a transformer-based architecture to attend to relevant BEV regions.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for BEV features."""
    
    def __init__(self, channels: int, max_h: int = 256, max_w: int = 256):
        super().__init__()
        
        # Create positional encodings
        pe_h = torch.zeros(max_h, channels // 2)
        pe_w = torch.zeros(max_w, channels // 2)
        
        position_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        position_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, channels // 2, 2).float() * (-math.log(10000.0) / (channels // 2))
        )
        
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        self.register_buffer("pe_h", pe_h)
        self.register_buffer("pe_w", pe_w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to features.
        
        Args:
            x: [B, C, H, W] feature map
            
        Returns:
            [B, C, H, W] features with positional encoding
        """
        B, C, H, W = x.shape
        
        # Get positional encodings
        pe_h = self.pe_h[:H, :].unsqueeze(1).expand(-1, W, -1)  # [H, W, C/2]
        pe_w = self.pe_w[:W, :].unsqueeze(0).expand(H, -1, -1)  # [H, W, C/2]
        
        pe = torch.cat([pe_h, pe_w], dim=-1)  # [H, W, C]
        pe = pe.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        return x + pe


class CrossAttention(nn.Module):
    """Cross-attention between query and key-value features."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attention forward.
        
        Args:
            query: [B, N, D] query features
            key: [B, M, D] key features
            value: [B, M, D] value features
            
        Returns:
            [B, N, D] attended features
        """
        B, N, D = query.shape
        M = key.shape[1]
        
        # Project
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        
        return out


class WaypointTransformerLayer(nn.Module):
    """Transformer layer for waypoint prediction."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: [B, N, D] waypoint queries
            context: [B, M, D] BEV context features
            
        Returns:
            [B, N, D] updated queries
        """
        # Cross-attention
        query = query + self.cross_attn(self.norm1(query), context, context)
        
        # MLP
        query = query + self.mlp(self.norm2(query))
        
        return query


class WaypointHead(nn.Module):
    """
    Waypoint prediction head for ego planning.
    
    Architecture:
    1. Pool BEV features around ego (center of BEV grid)
    2. Use learnable waypoint queries (one per waypoint)
    3. Cross-attend queries to BEV features
    4. Regress (x, y, heading) for each waypoint
    
    Waypoints are in ego frame at current timestamp.
    Horizons are configurable (e.g., 0.5s, 1.0s, ..., 5.0s).
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_waypoints: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 3,
        bev_size: Tuple[int, int] = (300, 300),
        ego_pool_size: int = 32,
    ):
        """
        Initialize waypoint head.
        
        Args:
            in_channels: Input BEV feature channels
            num_waypoints: Number of future waypoints to predict
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            bev_size: (H, W) BEV grid size
            ego_pool_size: Size of ego-centric pooling region
        """
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.hidden_dim = hidden_dim
        self.bev_size = bev_size
        self.ego_pool_size = ego_pool_size
        
        # Project BEV features
        self.bev_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding2D(hidden_dim, bev_size[0], bev_size[1])
        
        # Learnable waypoint queries
        self.waypoint_queries = nn.Parameter(torch.randn(num_waypoints, hidden_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            WaypointTransformerLayer(hidden_dim, num_heads=8, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # (x, y)
        )
        
        self.heading_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # (sin, cos)
        )
    
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Predict future ego waypoints.
        
        Args:
            bev_features: [B, C, H, W] BEV feature map
            
        Returns:
            [B, K, 3] waypoints (x, y, heading) in meters/radians
        """
        B = bev_features.shape[0]
        device = bev_features.device
        
        # Project BEV features
        bev = self.bev_proj(bev_features)
        
        # Add positional encoding
        bev = self.pos_enc(bev)
        
        # Pool around ego (center of BEV)
        H, W = bev.shape[2:]
        cx, cy = W // 2, H // 2
        ps = self.ego_pool_size // 2
        
        # Clamp to valid range
        x1, x2 = max(0, cx - ps), min(W, cx + ps)
        y1, y2 = max(0, cy - ps), min(H, cy + ps)
        
        ego_features = bev[:, :, y1:y2, x1:x2]  # [B, C, ego_h, ego_w]
        
        # Flatten to sequence
        ego_flat = ego_features.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Expand queries for batch
        queries = self.waypoint_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, C]
        
        # Apply transformer layers
        for layer in self.layers:
            queries = layer(queries, ego_flat)
        
        # Predict outputs
        positions = self.position_head(queries)  # [B, K, 2]
        headings_raw = self.heading_head(queries)  # [B, K, 2]
        
        # Normalize heading to unit vector and convert to angle
        headings_norm = F.normalize(headings_raw, dim=-1)
        headings = torch.atan2(headings_norm[..., 0], headings_norm[..., 1])  # [B, K]
        
        # Combine outputs
        waypoints = torch.cat([positions, headings.unsqueeze(-1)], dim=-1)  # [B, K, 3]
        
        return waypoints


class WaypointMLPHead(nn.Module):
    """
    Simple MLP-based waypoint head.
    
    Faster than transformer but less expressive.
    Good for lightweight deployment.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_waypoints: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 3,
        bev_size: Tuple[int, int] = (300, 300),
        ego_pool_size: int = 32,
    ):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.ego_pool_size = ego_pool_size
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((ego_pool_size, ego_pool_size))
        
        # Flatten and project
        input_dim = in_channels * ego_pool_size * ego_pool_size
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
        
        # Output: K waypoints x 3 (x, y, heading)
        layers.append(nn.Linear(hidden_dim, num_waypoints * 3))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        Predict future ego waypoints.
        
        Args:
            bev_features: [B, C, H, W] BEV feature map
            
        Returns:
            [B, K, 3] waypoints (x, y, heading)
        """
        B = bev_features.shape[0]
        
        # Pool around ego (center of BEV)
        H, W = bev_features.shape[2:]
        cx, cy = W // 2, H // 2
        ps = min(self.ego_pool_size, min(H, W) // 2)
        
        x1, x2 = max(0, cx - ps), min(W, cx + ps)
        y1, y2 = max(0, cy - ps), min(H, cy + ps)
        
        ego_features = bev_features[:, :, y1:y2, x1:x2]
        
        # Adaptive pool and flatten
        pooled = self.pool(ego_features)
        flat = pooled.flatten(1)
        
        # MLP
        out = self.mlp(flat)
        
        # Reshape to waypoints
        waypoints = out.view(B, self.num_waypoints, 3)
        
        return waypoints
