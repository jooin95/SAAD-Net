#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DefectLoCK-DDS: Deep-stage Decomposition Strategy for Industrial Defect Detection
==================================================================================

Inspired by OverLoCK (CVPR 2025), we adapt the "Overview-first-Look-Closely-next"
principle specifically for industrial defect classification.

Key Insight:
- Human inspectors first scan the entire image for anomalies (overview)
- Then focus on suspicious regions for detailed analysis (look closely)
- This is exactly how DDS works!

Architecture:
    Input Image (1024x1024)
           ↓
    ┌─────────────────────────────────────┐
    │         Base-Net (Swin V2)          │  ← Pretrained, mid-level features
    │         Stages 1-2: H/4 → H/16      │
    └─────────────────────────────────────┘
           ↓
    Mid-level Features (H/16 × W/16)
           ↓
    ┌──────────────────┐     ┌──────────────────┐
    │   Overview-Net   │     │    Focus-Net     │
    │   (Lightweight)  │────▶│    (Deep)        │
    │   Global Context │     │   Fine-grained   │
    │   Defect Prior   │     │   + Guidance     │
    └──────────────────┘     └──────────────────┘
           ↓                        ↓
    Auxiliary Loss           Main Classification
    (Coarse)                 (Fine-grained)

Paper Contributions:
1. DDS for Defect Detection: First application of DDS to industrial inspection
2. Defect-aware Overview-Net: Generates anomaly attention map
3. Guided Focus-Net: Uses overview guidance for better defect localization
4. Progressive CDL: Contrastive learning with warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict
import math

try:
    from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
    from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
    SWIN_AVAILABLE = True
except ImportError:
    SWIN_AVAILABLE = False


# ============================================================================
# 1. Base Modules (Proven Effective)
# ============================================================================

class ECABlock(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels: int):
        super().__init__()
        k_size = int(abs((math.log2(channels) + 1) / 2))
        k_size = k_size if k_size % 2 else k_size + 1
        k_size = max(3, k_size)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    """Spatial Attention"""
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
    """CBAM: Channel + Spatial Attention"""
    def __init__(self, channels: int):
        super().__init__()
        self.eca = ECABlock(channels)
        self.spatial = SpatialAttention()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.eca(x)
        x = self.spatial(x)
        return x


# ============================================================================
# 2. Overview-Net: Lightweight Global Context for Defect Detection
# ============================================================================

class DefectOverviewNet(nn.Module):
    """
    Lightweight Overview Network for Defect Detection
    
    Purpose: Quickly generate a coarse but semantically meaningful
    defect attention map that guides the Focus-Net.
    
    Design Principles:
    - Very lightweight (few parameters)
    - Fast inference
    - Produces "defect prior" for top-down guidance
    """
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()
        
        hidden_dim = in_channels // 2
        
        # Quick downsampling to get global view
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Global context aggregation
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Defect attention generator (spatial)
        self.defect_attention = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature projection for guidance
        self.guidance_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Auxiliary classifier (coarse classification)
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Mid-level features from Base-Net (B, C, H, W)
        
        Returns:
            dict with:
            - guidance: Top-down guidance features (B, C', H', W')
            - attention: Defect attention map (B, 1, H', W')
            - aux_logits: Auxiliary classification logits (B, num_classes)
            - context: Global context vector (B, hidden_dim)
        """
        # Downsample for overview
        feat = self.downsample(x)  # (B, hidden, H/4, W/4)
        
        # Global context
        context = self.global_context(feat)  # (B, hidden)
        
        # Defect attention map
        attention = self.defect_attention(feat)  # (B, 1, H/4, W/4)
        
        # Guidance features (attention-weighted)
        attended_feat = feat * (1 + attention)
        guidance = self.guidance_proj(attended_feat)  # (B, out_channels, H/4, W/4)
        
        # Auxiliary logits
        aux_logits = self.aux_classifier(feat)
        
        return {
            'guidance': guidance,
            'attention': attention,
            'aux_logits': aux_logits,
            'context': context
        }


# ============================================================================
# 3. Focus-Net: Fine-grained Analysis with Top-down Guidance
# ============================================================================

class GuidedFocusBlock(nn.Module):
    """
    Focus Block with Top-down Guidance from Overview-Net
    
    Key Innovation: The guidance from Overview-Net modulates
    both the feature extraction and attention computation.
    """
    def __init__(self, channels: int, guidance_channels: int):
        super().__init__()
        
        # Guidance integration (from Overview-Net)
        self.guidance_gate = nn.Sequential(
            nn.Conv2d(guidance_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Multi-scale local enhancement (proven effective)
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels // 4),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(channels // 4),
            nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor, guidance: Tensor = None) -> Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            guidance: Top-down guidance from Overview-Net (B, C', H', W')
        """
        B, C, H, W = x.shape
        
        # Apply guidance modulation if available
        if guidance is not None:
            # Upsample guidance to match x resolution
            guidance_up = F.interpolate(guidance, size=(H, W), mode='bilinear', align_corners=False)
            gate = self.guidance_gate(guidance_up)
            x = x * gate  # Modulate features with overview guidance
        
        # Multi-scale processing
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = F.interpolate(self.branch4(x), size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        
        return x + self.gamma * out


class DefectFocusNet(nn.Module):
    """
    Focus Network for Fine-grained Defect Analysis
    
    Receives mid-level features and top-down guidance from Overview-Net.
    Performs detailed defect feature extraction.
    """
    def __init__(self, in_channels: int, guidance_channels: int, num_blocks: int = 2):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        # Guided focus blocks
        self.focus_blocks = nn.ModuleList([
            GuidedFocusBlock(in_channels, guidance_channels)
            for _ in range(num_blocks)
        ])
        
        # CBAM for final attention
        self.cbam = CBAM(in_channels)
        
        # Small object attention (for tiny defects)
        self.small_obj_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor, guidance: Tensor = None) -> Tensor:
        """
        Args:
            x: Mid-level features (B, C, H, W)
            guidance: Top-down guidance from Overview-Net
        
        Returns:
            Enhanced features for classification
        """
        x = self.input_proj(x)
        
        # Apply guided focus blocks
        for block in self.focus_blocks:
            x = block(x, guidance)
        
        # CBAM attention
        x = self.cbam(x)
        
        # Small object attention
        small_attn = self.small_obj_attn(x)
        x = x * small_attn
        
        return x


# ============================================================================
# 4. Contrastive Learning Components
# ============================================================================

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=1)


# ============================================================================
# 5. Main Model: DefectLoCK-DDS
# ============================================================================

class DefectLoCKDDS(nn.Module):
    """
    DefectLoCK with Deep-stage Decomposition Strategy
    
    Architecture inspired by OverLoCK (CVPR 2025), adapted for defect detection.
    
    Key Components:
    1. Base-Net: Pretrained Swin V2 (Stages 1-2) for mid-level features
    2. Overview-Net: Lightweight network for global defect context
    3. Focus-Net: Deep network with top-down guidance for fine-grained analysis
    4. Contrastive Head: For supervised contrastive learning
    
    Training Strategy:
    - Joint training with auxiliary loss from Overview-Net
    - Progressive CDL warmup
    - Main loss from Focus-Net classification
    """
    def __init__(
        self,
        num_classes: int = 7,
        model_size: str = 't',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_contrastive: bool = True,
        contrastive_dim: int = 128,
        num_focus_blocks: int = 2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_contrastive = use_contrastive
        
        # ==================== Base-Net (Pretrained Swin V2) ====================
        if model_size == 't':
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            backbone = swin_v2_t(weights=weights)
            self.feat_dim = 768
        elif model_size == 's':
            weights = Swin_V2_S_Weights.DEFAULT if pretrained else None
            backbone = swin_v2_s(weights=weights)
            self.feat_dim = 768
        elif model_size == 'b':
            weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
            backbone = swin_v2_b(weights=weights)
            self.feat_dim = 1024
        else:
            raise ValueError(f"Unknown model_size: {model_size}")
        
        # Use full backbone as Base-Net
        self.base_net = backbone.features
        self.base_norm = backbone.norm
        self.base_permute = backbone.permute
        
        # ==================== Overview-Net ====================
        guidance_dim = self.feat_dim // 2
        self.overview_net = DefectOverviewNet(
            in_channels=self.feat_dim,
            out_channels=guidance_dim,
            num_classes=num_classes
        )
        
        # ==================== Focus-Net ====================
        self.focus_net = DefectFocusNet(
            in_channels=self.feat_dim,
            guidance_channels=guidance_dim,
            num_blocks=num_focus_blocks
        )
        
        # ==================== Classification Head ====================
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        # ==================== Contrastive Head ====================
        if use_contrastive:
            self.projection = ProjectionHead(self.feat_dim, 512, contrastive_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass with DDS strategy
        
        Returns:
            dict with:
            - logits: Main classification logits (B, num_classes)
            - aux_logits: Auxiliary logits from Overview-Net (B, num_classes)
            - embeddings: Contrastive embeddings (B, contrastive_dim)
            - defect_attention: Defect attention map from Overview-Net
            - features: Final feature vector
        """
        # ==================== Base-Net ====================
        # Extract mid-level features
        base_feat = self.base_net(x)
        base_feat = self.base_norm(base_feat)
        base_feat = self.base_permute(base_feat)  # (B, C, H, W)
        
        # ==================== Overview-Net (Overview First) ====================
        overview_out = self.overview_net(base_feat)
        guidance = overview_out['guidance']
        aux_logits = overview_out['aux_logits']
        defect_attention = overview_out['attention']
        
        # ==================== Focus-Net (Look Closely Next) ====================
        focus_feat = self.focus_net(base_feat, guidance)
        
        # Global pooling
        features = F.adaptive_avg_pool2d(focus_feat, 1).flatten(1)
        
        # ==================== Classification ====================
        logits = self.classifier(features)
        
        # ==================== Output ====================
        outputs = {
            'logits': logits,
            'aux_logits': aux_logits,
            'features': features,
            'defect_attention': defect_attention
        }
        
        if self.use_contrastive:
            outputs['embeddings'] = self.projection(features)
        
        return outputs


# ============================================================================
# 6. Loss Function
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self
        
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        mask_pos_sum = mask_pos.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / mask_pos_sum
        
        return -mean_log_prob_pos.mean()


class DefectLoCKDDSLoss(nn.Module):
    """
    Loss function for DefectLoCK-DDS
    
    Components:
    1. Main Focal Loss (from Focus-Net)
    2. Auxiliary Loss (from Overview-Net, weighted)
    3. Contrastive Loss (progressive warmup)
    """
    def __init__(
        self,
        num_classes: int = 7,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        aux_weight: float = 0.4,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.aux_weight = aux_weight
        self.contrastive_weight = contrastive_weight
        
        self.contrastive_loss = SupervisedContrastiveLoss(temperature)
        
        # Progressive warmup
        self.current_epoch = 0
        self.warmup_epochs = 10
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
    
    def get_progressive_weight(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            return self.contrastive_weight * (self.current_epoch / self.warmup_epochs)
        return self.contrastive_weight
    
    def focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.focal_gamma) * ce).mean()
    
    def forward(self, outputs: Dict[str, Tensor], targets: Tensor) -> Dict[str, Tensor]:
        # Main focal loss (Focus-Net)
        loss_main = self.focal_loss(outputs['logits'], targets)
        
        # Auxiliary loss (Overview-Net)
        loss_aux = self.focal_loss(outputs['aux_logits'], targets)
        
        # Contrastive loss (progressive)
        loss_contrastive = torch.tensor(0.0, device=loss_main.device)
        if 'embeddings' in outputs:
            progressive_weight = self.get_progressive_weight()
            if progressive_weight > 0:
                loss_contrastive = self.contrastive_loss(outputs['embeddings'], targets)
        
        # Total loss
        progressive_weight = self.get_progressive_weight()
        total = loss_main + self.aux_weight * loss_aux + progressive_weight * loss_contrastive
        
        return {
            'total': total,
            'main': loss_main,
            'aux': loss_aux,
            'contrastive': loss_contrastive,
            'contrastive_weight': torch.tensor(progressive_weight)
        }


# ============================================================================
# 7. Factory Functions
# ============================================================================

def get_defect_lock_dds(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> DefectLoCKDDS:
    """
    Factory function for DefectLoCK-DDS
    
    Configs:
    - 'default': Standard DDS configuration
    - 'light': Fewer focus blocks
    - 'deep': More focus blocks
    """
    configs = {
        'default': {
            'dropout': 0.3,
            'use_contrastive': True,
            'contrastive_dim': 128,
            'num_focus_blocks': 2
        },
        'light': {
            'dropout': 0.3,
            'use_contrastive': True,
            'contrastive_dim': 128,
            'num_focus_blocks': 1
        },
        'deep': {
            'dropout': 0.2,
            'use_contrastive': True,
            'contrastive_dim': 256,
            'num_focus_blocks': 3
        }
    }
    
    cfg = configs.get(config, configs['default'])
    cfg.update(kwargs)
    
    return DefectLoCKDDS(num_classes=num_classes, model_size=model_size, **cfg)


class DefectLoCKDDSWrapper(nn.Module):
    """Wrapper for train_unified.py compatibility"""
    def __init__(self, model: DefectLoCKDDS):
        super().__init__()
        self.model = model
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.model(x)


def get_defect_lock_dds_for_training(
    num_classes: int = 7,
    model_size: str = 't',
    **kwargs
) -> DefectLoCKDDSWrapper:
    model = get_defect_lock_dds(num_classes, model_size, **kwargs)
    return DefectLoCKDDSWrapper(model)


# ============================================================================
# 8. Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing DefectLoCK-DDS")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config in ['default', 'light', 'deep']:
        print(f"\nConfig: {config}")
        model = get_defect_lock_dds(num_classes=7, model_size='t', config=config)
        model = model.to(device)
        
        # Count parameters
        total = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Total: {total:.2f}M, Trainable: {trainable:.2f}M")
        
        # Test forward
        x = torch.randn(4, 3, 512, 512).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        print(f"  Main logits: {outputs['logits'].shape}")
        print(f"  Aux logits: {outputs['aux_logits'].shape}")
        print(f"  Defect attention: {outputs['defect_attention'].shape}")
        if 'embeddings' in outputs:
            print(f"  Embeddings: {outputs['embeddings'].shape}")
        
        # Test loss
        criterion = DefectLoCKDDSLoss(aux_weight=0.4, contrastive_weight=0.1)
        criterion.set_epoch(15)
        losses = criterion(outputs, labels)
        print(f"  Loss: {losses['total']:.4f} (main={losses['main']:.4f}, "
              f"aux={losses['aux']:.4f}, contrastive={losses['contrastive']:.4f})")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
