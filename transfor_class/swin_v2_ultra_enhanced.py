#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swin V2 Ultra Enhanced - Stable Version
Based on working swin_v2_enhanced structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    swin_v2_t, swin_v2_s, swin_v2_b,
    Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
)
from typing import Optional
import math
import numpy as np


# ============================================================================
# Attention Modules (Same as swin_v2_enhanced)
# ============================================================================

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Channel attention
        avg_out = self.channel_fc(self.channel_avg(x).view(B, C))
        max_out = self.channel_fc(self.channel_max(x).view(B, C))
        channel_weight = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        x = x * channel_weight
        
        # Spatial attention
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.max(dim=1, keepdim=True)[0]
        spatial_weight = torch.sigmoid(self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        
        return x * spatial_weight


class LocalEnhancementModule(nn.Module):
    """Multi-scale local feature enhancement - Same as swin_v2_enhanced"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv3x3 = nn.Conv2d(channels, channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(channels, channels // 4, 7, padding=3)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        f7 = self.conv7x7(x)
        
        multi_scale = torch.cat([f1, f3, f5, f7], dim=1)
        enhanced = self.fusion(multi_scale)
        
        return x + self.gamma * enhanced


# ============================================================================
# Loss Function
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - Same as original"""
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================================================
# Main Model - Same structure as swin_v2_enhanced
# ============================================================================

class SwinV2UltraEnhanced(nn.Module):
    """
    Swin V2 Ultra Enhanced
    Same structure as working swin_v2_enhanced
    """
    def __init__(
        self,
        num_classes: int,
        model_size: str = 't',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_cbam: bool = True,
        use_local_enhance: bool = True,
        **kwargs  # Accept and ignore extra kwargs
    ):
        super().__init__()
        
        self.use_cbam = use_cbam
        self.use_local_enhance = use_local_enhance
        
        # Backbone
        if model_size == 't':
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_t(weights=weights)
            self.feature_dim = 768
        elif model_size == 's':
            weights = Swin_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_s(weights=weights)
            self.feature_dim = 768
        else:
            weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_b(weights=weights)
            self.feature_dim = 1024
        
        self.backbone.head = nn.Identity()
        
        # Enhancement modules
        if use_cbam:
            self.cbam = CBAM(self.feature_dim, reduction=16)
        
        if use_local_enhance:
            self.local_enhance = LocalEnhancementModule(self.feature_dim)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head - Same as swin_v2_enhanced
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Backbone forward
        x = self.backbone.features(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # Enhancement
        if self.use_cbam:
            x = self.cbam(x)
        
        if self.use_local_enhance:
            x = self.local_enhance(x)
        
        # Pool and classify
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        
        return x


# ============================================================================
# Training Utilities
# ============================================================================

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.model = model
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization"""
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step(closure)
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def mixup_data(x, y, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    x_clone = x.clone()
    x_clone[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x_clone, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Factory Function
# ============================================================================

def get_swin_v2_ultra(
    num_classes: int,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> nn.Module:
    """
    Factory function
    
    Configs:
    - 'default': CBAM + LocalEnhancement (same as swin_v2_enhanced)
    - 'cbam': CBAM only
    - 'local': LocalEnhancement only
    - 'minimal': No enhancement
    """
    configs = {
        'default': {'use_cbam': True, 'use_local_enhance': True},
        'cbam': {'use_cbam': True, 'use_local_enhance': False},
        'local': {'use_cbam': False, 'use_local_enhance': True},
        'minimal': {'use_cbam': False, 'use_local_enhance': False},
        # Backward compatibility
        'fpn': {'use_cbam': True, 'use_local_enhance': True},
        'deep': {'use_cbam': True, 'use_local_enhance': True},
        'arcface': {'use_cbam': True, 'use_local_enhance': True},
        'full': {'use_cbam': True, 'use_local_enhance': True},
    }
    
    cfg = configs.get(config, configs['default'])
    cfg.update(kwargs)
    
    return SwinV2UltraEnhanced(
        num_classes=num_classes,
        model_size=model_size,
        **cfg
    )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Swin V2 Ultra Enhanced Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test model
    model = get_swin_v2_ultra(num_classes=7, model_size='t', config='default')
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # Test forward
    x = torch.randn(2, 3, 1024, 1024).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    print("=" * 60)
    print("All tests passed!")