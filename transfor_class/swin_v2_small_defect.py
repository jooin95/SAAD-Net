# -*- coding: utf-8 -*-
"""
Swin V2 Small Defect Detector V2
================================

Based on working swin_v2_enhanced structure with additional modules for small defect detection:
1. Multi-scale Local Enhancement (dilated convolutions)
2. ECA (Efficient Channel Attention) - lightweight alternative to SE
3. Spatial Attention for small objects
4. ASPP-style multi-scale feature extraction

This version removes the problematic FPN and uses the proven swin_v2_enhanced backbone structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
import math

try:
    from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
    from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
    SWIN_V2_AVAILABLE = True
except ImportError:
    SWIN_V2_AVAILABLE = False
    print("Warning: torchvision Swin V2 not available")


# ============================================================================
# Attention Modules
# ============================================================================

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA-Net, CVPR 2020)
    Lightweight alternative to SE block using 1D convolution
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size based on channel count
        k_size = int(abs((math.log2(channels) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        k_size = max(3, k_size)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        # Global average pooling: (B, C, 1, 1)
        y = self.avg_pool(x)
        # Reshape for 1D conv: (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D conv for channel interaction
        y = self.conv(y)
        # Reshape back: (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Scale
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on defect regions"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (ECCV 2018)
    Combines channel and spatial attention
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel attention (using ECA instead of SE for efficiency)
        self.eca = ECABlock(channels)
        # Spatial attention
        self.spatial = SpatialAttention(kernel_size=7)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.eca(x)
        x = self.spatial(x)
        return x


# ============================================================================
# Multi-scale Feature Extraction
# ============================================================================

class MultiScaleLocalEnhancement(nn.Module):
    """
    Multi-scale Local Enhancement Module
    Uses dilated convolutions for multi-scale receptive field without resolution loss
    """
    def __init__(self, channels: int):
        super().__init__()
        branch_ch = channels // 4
        
        # Different dilation rates for multi-scale
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, dilation=1, groups=branch_ch),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=2, dilation=2, groups=branch_ch),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=4, dilation=4, groups=branch_ch),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        # Global context branch
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_ch * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        # Learnable scale
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Global branch needs upsampling
        b_global = self.branch_global(x)
        b_global = F.interpolate(b_global, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        out = torch.cat([b1, b2, b3, b_global], dim=1)
        out = self.fusion(out)
        
        return x + self.gamma * out


class SmallObjectAttention(nn.Module):
    """
    Attention module specifically designed for small objects/defects
    Combines multi-scale spatial attention with channel attention
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # Channel attention
        self.eca = ECABlock(channels)
        
        # Multi-scale spatial attention with different kernel sizes
        # Smaller kernels for small defects
        self.spatial_1x1 = nn.Conv2d(channels, channels // 4, 1)
        self.spatial_3x3 = nn.Conv2d(channels, channels // 4, 3, padding=1, groups=channels // 4)
        self.spatial_5x5 = nn.Conv2d(channels, channels // 4, 5, padding=2, groups=channels // 4)
        self.spatial_7x7 = nn.Conv2d(channels, channels // 4, 7, padding=3, groups=channels // 4)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # Learnable scale
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor) -> Tensor:
        # Channel attention first
        x = self.eca(x)
        
        # Multi-scale spatial features
        s1 = self.spatial_1x1(x)
        s3 = self.spatial_3x3(x)
        s5 = self.spatial_5x5(x)
        s7 = self.spatial_7x7(x)
        
        # Concatenate and create attention
        spatial = torch.cat([s1, s3, s5, s7], dim=1)
        attention = self.fusion(spatial)
        
        return x + self.gamma * (x * attention)


# ============================================================================
# Main Model
# ============================================================================

class SwinV2SmallDefectV2(nn.Module):
    """
    Swin V2 Small Defect Detector V2
    
    Based on proven swin_v2_enhanced structure with additional modules:
    - CBAM attention
    - Multi-scale Local Enhancement (dilated convolutions)
    - Small Object Attention
    
    This version uses the standard Swin V2 output (no FPN) to ensure stability.
    """
    def __init__(
        self,
        num_classes: int = 7,
        model_size: str = 't',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_cbam: bool = True,
        use_local_enhance: bool = True,
        use_small_obj_attn: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_cbam = use_cbam
        self.use_local_enhance = use_local_enhance
        self.use_small_obj_attn = use_small_obj_attn
        
        # Load Swin V2 backbone
        if model_size == 't':
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_t(weights=weights)
            self.feat_dim = 768
        elif model_size == 's':
            weights = Swin_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_s(weights=weights)
            self.feat_dim = 768
        elif model_size == 'b':
            weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_b(weights=weights)
            self.feat_dim = 1024
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Remove original classifier
        self.backbone.head = nn.Identity()
        
        # Feature enhancement modules
        if use_cbam:
            self.cbam = CBAM(self.feat_dim)
        
        if use_local_enhance:
            self.local_enhance = MultiScaleLocalEnhancement(self.feat_dim)
        
        if use_small_obj_attn:
            self.small_obj_attn = SmallObjectAttention(self.feat_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize head
        self._init_head()
    
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features from backbone"""
        # Swin V2 forward (returns features before head)
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        x = self.backbone.permute(x)  # (B, H, W, C) -> (B, C, H, W)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        # Extract features from backbone
        x = self.forward_features(x)  # (B, C, H, W)
        
        # Apply enhancement modules
        if self.use_cbam:
            x = self.cbam(x)
        
        if self.use_local_enhance:
            x = self.local_enhance(x)
        
        if self.use_small_obj_attn:
            x = self.small_obj_attn(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)  # (B, C)
        
        # Classification
        x = self.head(x)
        
        return x


# ============================================================================
# Loss Function
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma: float = 2.0, alpha: float = None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        return focal_loss.mean()


# ============================================================================
# Factory Function
# ============================================================================

def get_swin_v2_small_defect_v2(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> SwinV2SmallDefectV2:
    """
    Factory function for SwinV2SmallDefectV2
    
    Args:
        num_classes: Number of output classes
        model_size: 't' (tiny), 's' (small), 'b' (base)
        config: Configuration preset
            - 'default': All modules enabled (recommended)
            - 'light': Only CBAM (fastest)
            - 'full': All modules + lower dropout
    
    Returns:
        SwinV2SmallDefectV2 model
    """
    configs = {
        'default': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'dropout': 0.3
        },
        'light': {
            'use_cbam': True,
            'use_local_enhance': False,
            'use_small_obj_attn': False,
            'dropout': 0.3
        },
        'full': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'dropout': 0.2
        }
    }
    
    if config not in configs:
        print(f"Warning: Unknown config '{config}', using 'default'")
        config = 'default'
    
    cfg = configs[config]
    cfg.update(kwargs)
    
    return SwinV2SmallDefectV2(
        num_classes=num_classes,
        model_size=model_size,
        **cfg
    )


# Backward compatibility alias
def get_swin_v2_small_defect(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> SwinV2SmallDefectV2:
    """Alias for backward compatibility"""
    return get_swin_v2_small_defect_v2(num_classes, model_size, config, **kwargs)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing SwinV2SmallDefectV2")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test model creation
    for config in ['default', 'light', 'full']:
        print(f"\nConfig: {config}")
        model = get_swin_v2_small_defect_v2(num_classes=7, model_size='t', config=config)
        model = model.to(device)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Parameters: {params:.2f}M")
        
        # Test forward pass
        x = torch.randn(2, 3, 512, 512).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"Input: {x.shape} -> Output: {out.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)