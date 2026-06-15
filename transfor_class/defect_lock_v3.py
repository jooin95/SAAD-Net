#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DefectLoCK V3: Defect-aware Classification with Contrastive Learning
=====================================================================

Based on proven swin_v2_enhanced structure (80%+) with paper contributions:

1. PROVEN MODULES (from swin_v2_enhanced):
   - Swin V2 Backbone (Pretrained)
   - CBAM (Channel + Spatial Attention)
   - MultiScaleLocalEnhancement (dilated convolutions)
   - SmallObjectAttention

2. PAPER CONTRIBUTIONS:
   - Supervised Contrastive Learning (CDL) for fine-grained defect separation
   - Class-balanced Focal Loss
   - Feature projection head for better embedding

Architecture:
    Input → Swin V2 Backbone → CBAM → MultiScale → SmallObjAttn 
                                                      ↓
                               Classification Head ← GAP
                                      ↓
                             Contrastive Head (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
import math

try:
    from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
    from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
    SWIN_V2_AVAILABLE = True
except ImportError:
    SWIN_V2_AVAILABLE = False
    print("Warning: torchvision Swin V2 not available")


# ============================================================================
# 1. Proven Attention Modules (from swin_v2_enhanced)
# ============================================================================

class ECABlock(nn.Module):
    """Efficient Channel Attention (CVPR 2020)"""
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k_size = int(abs((math.log2(channels) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        k_size = max(3, k_size)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (ECCV 2018)"""
    def __init__(self, channels: int):
        super().__init__()
        self.eca = ECABlock(channels)
        self.spatial = SpatialAttention(kernel_size=7)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.eca(x)
        x = self.spatial(x)
        return x


# ============================================================================
# 2. Proven Multi-scale Modules (from swin_v2_enhanced)
# ============================================================================

class MultiScaleLocalEnhancement(nn.Module):
    """Multi-scale Local Enhancement with dilated convolutions"""
    def __init__(self, channels: int):
        super().__init__()
        branch_ch = channels // 4
        
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
        
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_ch * 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor) -> Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b_global = F.interpolate(self.branch_global(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = self.fusion(torch.cat([b1, b2, b3, b_global], dim=1))
        return x + self.gamma * out


class SmallObjectAttention(nn.Module):
    """Attention module for small defects"""
    def __init__(self, channels: int):
        super().__init__()
        self.eca = ECABlock(channels)
        
        # Multi-scale spatial attention
        self.spatial_1x1 = nn.Conv2d(channels, channels // 4, 1)
        self.spatial_3x3 = nn.Conv2d(channels, channels // 4, 3, padding=1, groups=channels // 4)
        self.spatial_5x5 = nn.Conv2d(channels, channels // 4, 5, padding=2, groups=channels // 4)
        self.spatial_7x7 = nn.Conv2d(channels, channels // 4, 7, padding=3, groups=channels // 4)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.eca(x)
        
        s1 = self.spatial_1x1(x)
        s3 = self.spatial_3x3(x)
        s5 = self.spatial_5x5(x)
        s7 = self.spatial_7x7(x)
        
        spatial = torch.cat([s1, s3, s5, s7], dim=1)
        attention = self.fusion(spatial)
        
        return x + self.gamma * (x * attention)


# ============================================================================
# 3. NEW: Contrastive Learning Head (Paper Contribution)
# ============================================================================

class ContrastiveHead(nn.Module):
    """
    Projection head for Supervised Contrastive Learning
    Maps features to a normalized embedding space for contrastive loss
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        z = self.projector(x)
        return F.normalize(z, dim=1)  # L2 normalize


# ============================================================================
# 4. Main Model: DefectLoCK V3
# ============================================================================

class DefectLoCKv3(nn.Module):
    """
    DefectLoCK V3: Defect-aware Classification with Contrastive Learning
    
    Combines proven modules with novel contrastive learning for defect detection.
    
    Args:
        num_classes: Number of defect classes
        model_size: 't' (tiny), 's' (small), 'b' (base)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
        use_cbam: Enable CBAM attention
        use_local_enhance: Enable multi-scale local enhancement
        use_small_obj_attn: Enable small object attention
        use_contrastive: Enable contrastive learning head
        contrastive_dim: Output dimension of contrastive head
    """
    def __init__(
        self,
        num_classes: int = 7,
        model_size: str = 't',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_cbam: bool = True,
        use_local_enhance: bool = True,
        use_small_obj_attn: bool = True,
        use_contrastive: bool = True,
        contrastive_dim: int = 128
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_cbam = use_cbam
        self.use_local_enhance = use_local_enhance
        self.use_small_obj_attn = use_small_obj_attn
        self.use_contrastive = use_contrastive
        
        # ==================== Backbone ====================
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
        
        # ==================== Feature Enhancement ====================
        if use_cbam:
            self.cbam = CBAM(self.feat_dim)
        
        if use_local_enhance:
            self.local_enhance = MultiScaleLocalEnhancement(self.feat_dim)
        
        if use_small_obj_attn:
            self.small_obj_attn = SmallObjectAttention(self.feat_dim)
        
        # ==================== Classification Head ====================
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # ==================== Contrastive Head (NEW) ====================
        if use_contrastive:
            self.contrastive_head = ContrastiveHead(
                in_dim=self.feat_dim,
                hidden_dim=512,
                out_dim=contrastive_dim
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features from backbone"""
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        x = self.backbone.permute(x)  # (B, H, W, C) -> (B, C, H, W)
        return x
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass
        
        Returns:
            dict with keys:
                - 'logits': Classification logits [B, num_classes]
                - 'embeddings': Contrastive embeddings [B, contrastive_dim] (if use_contrastive)
                - 'features': Raw features [B, feat_dim] (for analysis)
        """
        # Extract features
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
        features = x.flatten(1)  # (B, C)
        
        # Classification
        logits = self.head(features)
        
        # Output dict
        output = {
            'logits': logits,
            'features': features
        }
        
        # Contrastive embeddings
        if self.use_contrastive:
            embeddings = self.contrastive_head(features)
            output['embeddings'] = embeddings
        
        return output


# ============================================================================
# 5. NEW: Loss Functions (Paper Contribution)
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (NeurIPS 2020)
    Pulls together samples of the same class, pushes apart different classes
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings [B, D]
            labels: Class labels [B]
        Returns:
            Contrastive loss scalar
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        mask_no_diag = mask - torch.eye(batch_size, device=device)
        
        # For numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Compute log-softmax
        exp_logits = torch.exp(logits)
        
        # Mask out self-contrast
        mask_self = torch.ones_like(similarity) - torch.eye(batch_size, device=device)
        exp_logits = exp_logits * mask_self
        
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask_no_diag
        num_pos_pairs = mask_pos_pairs.sum(dim=1)
        
        # Avoid division by zero
        num_pos_pairs = torch.clamp(num_pos_pairs, min=1)
        
        mean_log_prob_pos = (mask_pos_pairs * log_prob).sum(dim=1) / num_pos_pairs
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class DefectLoCKv3Loss(nn.Module):
    """
    Combined loss for DefectLoCK V3:
    - Focal Loss for classification (handles class imbalance)
    - Supervised Contrastive Loss (improves feature separation)
    """
    def __init__(
        self,
        num_classes: int = 7,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
    
    def focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Focal loss with label smoothing"""
        ce_loss = F.cross_entropy(
            logits, targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()
    
    def forward(
        self, 
        outputs: Dict[str, Tensor], 
        targets: Tensor
    ) -> Dict[str, Tensor]:
        """
        Args:
            outputs: Model outputs dict with 'logits' and optionally 'embeddings'
            targets: Ground truth labels [B]
        Returns:
            Dict with 'total', 'focal', 'contrastive' losses
        """
        logits = outputs['logits']
        
        # Focal loss
        loss_focal = self.focal_loss(logits, targets)
        
        # Contrastive loss
        loss_contrastive = torch.tensor(0.0, device=logits.device)
        if 'embeddings' in outputs and self.contrastive_weight > 0:
            loss_contrastive = self.contrastive_loss(outputs['embeddings'], targets)
        
        # Total loss
        total_loss = loss_focal + self.contrastive_weight * loss_contrastive
        
        return {
            'total': total_loss,
            'focal': loss_focal,
            'contrastive': loss_contrastive
        }


# ============================================================================
# 6. Factory Functions
# ============================================================================

def get_defect_lock_v3(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> DefectLoCKv3:
    """
    Factory function for DefectLoCK V3
    
    Args:
        num_classes: Number of classes
        model_size: 't', 's', 'b'
        config: 'default', 'light', 'full', 'contrastive_only'
    """
    configs = {
        'default': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'use_contrastive': True,
            'dropout': 0.3
        },
        'light': {
            'use_cbam': True,
            'use_local_enhance': False,
            'use_small_obj_attn': False,
            'use_contrastive': True,
            'dropout': 0.3
        },
        'full': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'use_contrastive': True,
            'dropout': 0.2
        },
        'no_contrastive': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'use_contrastive': False,
            'dropout': 0.3
        }
    }
    
    if config not in configs:
        print(f"Warning: Unknown config '{config}', using 'default'")
        config = 'default'
    
    cfg = configs[config]
    cfg.update(kwargs)
    
    return DefectLoCKv3(num_classes=num_classes, model_size=model_size, **cfg)


# ============================================================================
# 7. Wrapper for train_unified.py compatibility
# ============================================================================

class DefectLoCKv3Wrapper(nn.Module):
    """
    Wrapper to make DefectLoCK V3 compatible with train_unified.py
    Returns dict with 'logits' and 'embeddings' for contrastive loss
    """
    def __init__(self, model: DefectLoCKv3):
        super().__init__()
        self.model = model
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Return full dict for loss computation
        return self.model(x)


def get_defect_lock_v3_for_training(
    num_classes: int = 7,
    model_size: str = 't',
    **kwargs
) -> DefectLoCKv3Wrapper:
    """Get wrapped model for train_unified.py"""
    model = get_defect_lock_v3(num_classes=num_classes, model_size=model_size, **kwargs)
    return DefectLoCKv3Wrapper(model)


# ============================================================================
# 8. Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing DefectLoCK V3")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test model
    model = get_defect_lock_v3(num_classes=7, model_size='t', config='default')
    model = model.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    # Test forward
    x = torch.randn(4, 3, 512, 512).to(device)
    labels = torch.randint(0, 7, (4,)).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Logits: {outputs['logits'].shape}")
    print(f"Features: {outputs['features'].shape}")
    print(f"Embeddings: {outputs['embeddings'].shape}")
    
    # Test loss
    criterion = DefectLoCKv3Loss(num_classes=7, contrastive_weight=0.1)
    losses = criterion(outputs, labels)
    print(f"\nLosses:")
    print(f"  Total: {losses['total']:.4f}")
    print(f"  Focal: {losses['focal']:.4f}")
    print(f"  Contrastive: {losses['contrastive']:.4f}")
    
    # Test wrapper
    print("\nTesting wrapper...")
    wrapper = DefectLoCKv3Wrapper(model)
    logits = wrapper(x)
    print(f"Wrapper output: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
