#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DefectLoCK V3 Pro: Enhanced Version for Paper Publication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
import math

try:
    from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
    from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
    SWIN_AVAILABLE = True
except ImportError:
    SWIN_AVAILABLE = False


class ECABlock(nn.Module):
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
    def __init__(self, channels: int):
        super().__init__()
        self.eca = ECABlock(channels)
        self.spatial = SpatialAttention()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.eca(x)
        x = self.spatial(x)
        return x


class MultiScaleLocalEnhancement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels // 4, 3, padding=1, dilation=1),
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
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = F.interpolate(self.branch4(x), size=(H, W), mode='bilinear', align_corners=False)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        return x + self.gamma * out


class SmallObjectAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, k, padding=k // 2)
            for k in [1, 3, 5, 7]
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        feats = [branch(x) for branch in self.branches]
        combined = torch.cat(feats, dim=1)
        attention = self.fusion(combined)
        return x * attention


class ProjectionHead(nn.Module):
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
        x = self.net(x)
        return F.normalize(x, dim=1)


class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
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


class DefectLoCKv3Pro(nn.Module):
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
            raise ValueError(f"Unknown model_size: {model_size}")
        
        self.backbone.head = nn.Identity()
        
        if use_cbam:
            self.cbam = CBAM(self.feat_dim)
        
        if use_local_enhance:
            self.local_enhance = MultiScaleLocalEnhancement(self.feat_dim)
        
        if use_small_obj_attn:
            self.small_obj_attn = SmallObjectAttention(self.feat_dim)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        if use_contrastive:
            self.projection = ProjectionHead(self.feat_dim, 512, contrastive_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: Tensor) -> Tensor:
        x = self.backbone.features(x)
        x = self.backbone.norm(x)
        x = self.backbone.permute(x)
        return x
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.forward_features(x)
        
        if self.use_cbam:
            x = self.cbam(x)
        
        if self.use_local_enhance:
            x = self.local_enhance(x)
        
        if self.use_small_obj_attn:
            x = self.small_obj_attn(x)
        
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.classifier(features)
        
        outputs = {
            'logits': logits,
            'features': features
        }
        
        if self.use_contrastive:
            outputs['embeddings'] = self.projection(features)
        
        return outputs


class DefectLoCKv3ProLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 0.1,
        ortho_weight: float = 0.01,
        temperature: float = 0.07,
        hard_negative_weight: float = 0.5
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        
        self.contrastive_loss = HardNegativeContrastiveLoss(temperature=temperature)
        
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
        loss_focal = self.focal_loss(outputs['logits'], targets)
        
        loss_contrastive = torch.tensor(0.0, device=loss_focal.device)
        if 'embeddings' in outputs:
            progressive_weight = self.get_progressive_weight()
            if progressive_weight > 0:
                loss_contrastive = self.contrastive_loss(outputs['embeddings'], targets)
        
        progressive_weight = self.get_progressive_weight()
        total = loss_focal + progressive_weight * loss_contrastive
        
        return {
            'total': total,
            'focal': loss_focal,
            'contrastive': loss_contrastive,
            'contrastive_weight': torch.tensor(progressive_weight)
        }


def get_defect_lock_v3_pro(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> DefectLoCKv3Pro:
    configs = {
        'default': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'use_contrastive': True,
            'dropout': 0.3,
            'contrastive_dim': 128
        },
        'baseline': {
            'use_cbam': True,
            'use_local_enhance': True,
            'use_small_obj_attn': True,
            'use_contrastive': False,
            'dropout': 0.3
        }
    }
    
    cfg = configs.get(config, configs['default'])
    cfg.update(kwargs)
    
    return DefectLoCKv3Pro(num_classes=num_classes, model_size=model_size, **cfg)


class DefectLoCKv3ProWrapper(nn.Module):
    def __init__(self, model: DefectLoCKv3Pro):
        super().__init__()
        self.model = model
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.model(x)


def get_defect_lock_v3_pro_for_training(
    num_classes: int = 7,
    model_size: str = 't',
    **kwargs
) -> DefectLoCKv3ProWrapper:
    model = get_defect_lock_v3_pro(num_classes, model_size, **kwargs)
    return DefectLoCKv3ProWrapper(model)


if __name__ == '__main__':
    print("Testing DefectLoCK V3 Pro")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_defect_lock_v3_pro(num_classes=7, model_size='t')
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.2f}M")
    x = torch.randn(2, 3, 512, 512).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    print(f"Logits: {outputs['logits'].shape}")
    print("Test passed!")