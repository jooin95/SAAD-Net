#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DefectLoCK V2 Improved (SAAD-Net)
=================================
Paper-ready implementation with:
- Lightweight Anomaly-Aware Attention (LAA)
- Simplified Scale-Adaptive Attention (S-SAA)
- CBAM (Convolutional Block Attention Module)
- Supervised Contrastive Defect Learning (CDL) with Progressive Warmup

Supports:
- torchvision backend (256x256 only)
- timm backend (variable img_size: 256, 512, 1024, etc.)

Ablation Study Configs:
- baseline: Swin V2-T only
- cbam_only: + CBAM
- cbam_laa: + CBAM + LAA
- attention_only: + CBAM + LAA + S-SAA (no CDL)
- default: Full model (CBAM + LAA + S-SAA + CDL)
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
    SWIN_V2_AVAILABLE = True
except ImportError:
    SWIN_V2_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ============================================================================
# 1. Lightweight Anomaly-Aware Attention (LAA)
# ============================================================================

class LightweightAnomalyAttention(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # ==================== Channel Attention ====================
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP for channel-wise anomaly statistics
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # ==================== Spatial Anomaly Map ====================
        self.spatial_anomaly = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # ==================== Learnable Enhancement Factor ====================
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Channel attention
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        channel_stats = torch.cat([avg_feat, max_feat], dim=1)
        channel_att = self.channel_mlp(channel_stats)
        x_channel = x * channel_att
        
        # Spatial anomaly map
        anomaly_map = self.spatial_anomaly(x_channel)
        
        # Anomaly-aware enhancement
        enhanced = x_channel * (1 + self.alpha * anomaly_map)
        
        return enhanced, anomaly_map


# ============================================================================
# 2. Simplified Scale-Adaptive Attention (S-SAA)
# ============================================================================

class SimplifiedSAA(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        branch_ch = channels // 4
        self.branch_ch = branch_ch
        
        # Multi-Scale Branches
        self.branch_small = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        self.branch_medium = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        self.branch_large = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU(),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.GELU()
        )
        
        # Scale Predictor
        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )
        
        # Output Projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(branch_ch, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        # LayerScale-style channel-wise residual gate.
        # Initialized to a small NONZERO value (1e-1 scaled per channel) so the
        # S-SAA branch contributes from the first step and the scale_predictor
        # receives a real gradient, instead of collapsing to gamma~0.
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, C, H, W = x.shape

        f_small = self.branch_small(x)
        f_medium = self.branch_medium(x)
        f_large = self.branch_large(x)
        f_global = F.interpolate(
            self.branch_global(x),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        scale_weights = self.scale_predictor(x)

        stacked = torch.stack([f_small, f_medium, f_large, f_global], dim=1)
        w = scale_weights.view(B, 4, 1, 1, 1)
        weighted = (w * stacked).sum(dim=1)

        fused = self.out_conv(weighted)
        output = x + self.gamma.view(1, -1, 1, 1) * fused

        return output, scale_weights


# ============================================================================
# 3. CBAM (Convolutional Block Attention Module)
# ============================================================================

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
        attention = self.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ECABlock(channels)
        self.spatial_att = SpatialAttention(kernel_size=7)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ============================================================================
# 4. Contrastive Head for CDL
# ============================================================================

class ContrastiveHead(nn.Module):
    
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
        return F.normalize(self.projector(x), dim=1)


# ============================================================================
# 5. Main Model: DefectLoCK V2 Improved
# ============================================================================

class DefectLoCKv2Improved(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        model_size: str = 't',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_cbam: bool = True,
        use_anomaly: bool = True,
        use_saa: bool = True,
        use_contrastive: bool = True,
        freeze_backbone_stages: int = 0,
        img_size: int = 256,
        backend: str = 'auto',
        saa_before_laa: bool = False,
        scale_to_cdl: bool = False
    ):
        super().__init__()
        
        self.use_cbam = use_cbam
        self.use_anomaly = use_anomaly
        self.use_saa = use_saa
        self.use_contrastive = use_contrastive
        self.img_size = img_size
        # Method 3: run S-SAA before LAA (CBAM -> S-SAA -> LAA)
        self.saa_before_laa = saa_before_laa
        # Method 4: feed S-SAA scale summary into the contrastive (CDL) embedding
        self.scale_to_cdl = scale_to_cdl and use_saa and use_contrastive
        
        # ==================== Backend Selection ====================
        if backend == 'auto':
            if img_size != 256 and TIMM_AVAILABLE:
                backend = 'timm'
            elif SWIN_V2_AVAILABLE:
                backend = 'torchvision'
            elif TIMM_AVAILABLE:
                backend = 'timm'
            else:
                raise ImportError("Neither torchvision nor timm available!")
        
        self.backend = backend
        
        # ==================== Backbone ====================
        if backend == 'timm':
            self._build_timm_backbone(model_size, pretrained, img_size)
        else:
            self._build_torchvision_backbone(model_size, pretrained)
        
        # Freeze backbone stages if specified
        if freeze_backbone_stages > 0:
            self._freeze_backbone_stages(freeze_backbone_stages)
        
        # ==================== Enhancement Modules ====================
        if use_cbam:
            self.cbam = CBAM(self.feat_dim)
        
        if use_anomaly:
            self.anomaly_module = LightweightAnomalyAttention(self.feat_dim)
        
        if use_saa:
            self.saa = SimplifiedSAA(self.feat_dim)
        
        # ==================== Classification Head ====================
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        # ==================== Contrastive Head ====================
        if use_contrastive:
            if self.scale_to_cdl:
                # Method 4: expand the 4-dim scale_weights into a small embedding
                # and concatenate it with the GAP feature before the projection MLP,
                # so scale information explicitly shapes the contrastive space.
                self.scale_embed = nn.Sequential(
                    nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64)
                )
                self.contrastive_head = ContrastiveHead(self.feat_dim + 64, 512, 128)
            else:
                self.contrastive_head = ContrastiveHead(self.feat_dim, 512, 128)
        
        self._init_weights()
    
    def _build_timm_backbone(self, model_size: str, pretrained: bool, img_size: int):
        """Build backbone using timm (supports variable img_size)"""
        model_names = {
            't': 'swinv2_tiny_window16_256',
            's': 'swinv2_small_window16_256',
            'b': 'swinv2_base_window16_256'
        }
        feat_dims = {'t': 768, 's': 768, 'b': 1024}
        
        model_name = model_names.get(model_size, model_names['t'])
        self.feat_dim = feat_dims.get(model_size, 768)
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            img_size=img_size
        )
        
        print(f"[SAAD-Net] Using timm backbone: {model_name}, img_size={img_size}")
    
    def _build_torchvision_backbone(self, model_size: str, pretrained: bool):
        """Build backbone using torchvision (256 only)"""
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
        print(f"[SAAD-Net] Using torchvision backbone: swin_v2_{model_size}")
    
    def _freeze_backbone_stages(self, num_stages: int):
        """Freeze first N stages of backbone"""
        if self.backend == 'timm':
            if hasattr(self.backbone, 'layers'):
                for i, layer in enumerate(self.backbone.layers):
                    if i < num_stages:
                        for param in layer.parameters():
                            param.requires_grad = False
        else:
            for i, layer in enumerate(self.backbone.features):
                if i < num_stages * 2:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: Tensor) -> Tensor:
        """Extract features from backbone - returns (B, C, H, W)"""
        if self.backend == 'timm':
            # timm swinv2 forward_features returns (B, H*W, C)
            feat = self.backbone.forward_features(x)
            
            # timm version compatibility: may return (B,H,W,C) or (B,N,C)
            if feat.dim() == 4:
                # (B, H, W, C) -> (B, C, H, W)
                feat = feat.permute(0, 3, 1, 2).contiguous()
                return feat
            # (B, N, C) where N = H*W
            B, N, C = feat.shape
            H = W = int(math.sqrt(N))
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            return feat
        else:
            # torchvision
            x = self.backbone.features(x)
            x = self.backbone.norm(x)
            x = self.backbone.permute(x)  # (B, H, W, C) -> (B, C, H, W)
            return x
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # ==================== Backbone Features ====================
        x = self.forward_features(x)  # (B, C, H, W)
        
        # ==================== Enhancement Modules ====================
        anomaly_map = None
        scale_weights = None
        
        if self.use_cbam:
            x = self.cbam(x)
        
        if self.saa_before_laa:
            # Method 3: CBAM -> S-SAA -> LAA
            if self.use_saa:
                x, scale_weights = self.saa(x)
            if self.use_anomaly:
                x, anomaly_map = self.anomaly_module(x)
        else:
            # Default: CBAM -> LAA -> S-SAA
            if self.use_anomaly:
                x, anomaly_map = self.anomaly_module(x)
            if self.use_saa:
                x, scale_weights = self.saa(x)
        
        # ==================== Global Pooling ====================
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # ==================== Classification ====================
        logits = self.classifier(features)
        
        # ==================== Output Dictionary ====================
        outputs = {
            'logits': logits,
            'features': features
        }
        
        if self.use_contrastive:
            if self.scale_to_cdl and scale_weights is not None:
                # Method 4: inject scale information into the contrastive embedding
                scale_emb = self.scale_embed(scale_weights)
                embeddings = self.contrastive_head(torch.cat([features, scale_emb], dim=1))
            else:
                embeddings = self.contrastive_head(features)
            outputs['embeddings'] = embeddings
        
        if anomaly_map is not None:
            outputs['anomaly_map'] = anomaly_map
        
        if scale_weights is not None:
            outputs['scale_weights'] = scale_weights
        
        return outputs


# ============================================================================
# 6. Loss Functions
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):

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
        mask = torch.eq(labels, labels.T).float().to(device)
        mask_no_diag = mask - torch.eye(batch_size, device=device)
        
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        mask_self = torch.ones_like(similarity) - torch.eye(batch_size, device=device)
        exp_logits = exp_logits * mask_self
        
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        num_pos = mask_no_diag.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (mask_no_diag * log_prob).sum(dim=1) / num_pos
        
        return -mean_log_prob_pos.mean()


class DefectLoCKv2ImprovedLoss(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 7,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        scale_weight: float = 0.1,
        class_names: Optional[list] = None,
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.contrastive_weight = contrastive_weight
        self.scale_weight = scale_weight
        self.contrastive_loss = SupervisedContrastiveLoss(temperature)

        # ---- Scale-supervision target table ----
        # Branch order in S-SAA scale_predictor output: [small(d=1), medium(d=2),
        # large(d=4), global]. Each defect class is mapped to the branch(es) whose
        # receptive field matches its physical size, giving the scale_predictor an
        # explicit learning signal so S-SAA truly becomes scale-adaptive.
        #   small defects  (~0.4 mm): pinhole, dust, discoloration -> small branch
        #   large defects  (~3   mm): E_fold, G_fold, bubble        -> large branch
        #   good (no defect)                                        -> global/uniform
        self.register_buffer(
            'scale_target_table',
            self._build_scale_target_table(num_classes, class_names),
            persistent=False
        )

    @staticmethod
    def _build_scale_target_table(num_classes, class_names):
        # Default soft targets over [small, medium, large, global]
        small  = [0.70, 0.20, 0.05, 0.05]
        large  = [0.05, 0.20, 0.70, 0.05]
        good   = [0.25, 0.25, 0.25, 0.25]
        # Canonical PEMFC class order; fall back to uniform if names differ
        size_map = {
            'pinhole': small, 'dust': small, 'discoloration': small, 'discol': small,
            'e_fold': large, 'g_fold': large, 'bubble': large,
            'good': good,
        }
        table = torch.full((num_classes, 4), 0.25)
        if class_names is not None and len(class_names) == num_classes:
            for i, name in enumerate(class_names):
                key = str(name).strip().lower()
                if key in size_map:
                    table[i] = torch.tensor(size_map[key])
        return table

    def focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce = F.cross_entropy(
            logits, targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.focal_gamma
        return (focal_weight * ce).mean()
    
    def forward(
        self, 
        outputs: Dict[str, Tensor], 
        targets: Tensor,
        contrastive_weight: Optional[float] = None
    ) -> Dict[str, Tensor]:
        loss_focal = self.focal_loss(outputs['logits'], targets)
        
        weight = contrastive_weight if contrastive_weight is not None else self.contrastive_weight
        loss_contrastive = torch.tensor(0.0, device=loss_focal.device)
        
        if 'embeddings' in outputs and weight > 0:
            loss_contrastive = self.contrastive_loss(outputs['embeddings'], targets)

        # ---- Scale supervision (Method 2) ----
        loss_scale = torch.tensor(0.0, device=loss_focal.device)
        if 'scale_weights' in outputs and self.scale_weight > 0:
            pred_w = outputs['scale_weights'].clamp_min(1e-8)          # (B,4) softmax probs
            tgt_w  = self.scale_target_table.to(pred_w.device)[targets]  # (B,4)
            # KL(target || pred): pull predicted branch weights toward class-expected scale
            loss_scale = (tgt_w * (tgt_w.clamp_min(1e-8).log() - pred_w.log())).sum(dim=1).mean()

        total = loss_focal + weight * loss_contrastive + self.scale_weight * loss_scale
        
        return {
            'total': total,
            'focal': loss_focal,
            'contrastive': loss_contrastive,
            'scale': loss_scale
        }


# ============================================================================
# 7. Factory Functions
# ============================================================================

def get_defect_lock_v2_improved(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> DefectLoCKv2Improved:
    """Factory function for DefectLoCK V2 Improved"""
    configs = {
        'baseline': {
            'use_cbam': False, 'use_anomaly': False, 'use_saa': False,
            'use_contrastive': False, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'cbam_only': {
            'use_cbam': True, 'use_anomaly': False, 'use_saa': False,
            'use_contrastive': False, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'cbam_laa': {
            'use_cbam': True, 'use_anomaly': True, 'use_saa': False,
            'use_contrastive': False, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'attention_only': {
            'use_cbam': True, 'use_anomaly': True, 'use_saa': True,
            'use_contrastive': False, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'cbam_laa_cdl': {
            'use_cbam': True, 'use_anomaly': True, 'use_saa': False,
            'use_contrastive': True, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'default': {
            'use_cbam': True, 'use_anomaly': True, 'use_saa': True,
            'use_contrastive': True, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'default_saa_first': {
            # Method 3: CBAM -> S-SAA -> LAA
            'use_cbam': True, 'use_anomaly': True, 'use_saa': True,
            'use_contrastive': True, 'freeze_backbone_stages': 0, 'dropout': 0.3,
            'saa_before_laa': True
        },
        'default_scale_cdl': {
            # Method 3 + 4: reordered AND scale fed into CDL embedding
            'use_cbam': True, 'use_anomaly': True, 'use_saa': True,
            'use_contrastive': True, 'freeze_backbone_stages': 0, 'dropout': 0.3,
            'saa_before_laa': True, 'scale_to_cdl': True
        },
        'light': {
            'use_cbam': True, 'use_anomaly': False, 'use_saa': False,
            'use_contrastive': True, 'freeze_backbone_stages': 0, 'dropout': 0.3
        },
        'frozen': {
            'use_cbam': True, 'use_anomaly': True, 'use_saa': True,
            'use_contrastive': True, 'freeze_backbone_stages': 2, 'dropout': 0.4
        },
        'full': {
            'use_cbam': True, 'use_anomaly': True, 'use_saa': True,
            'use_contrastive': True, 'freeze_backbone_stages': 0, 'dropout': 0.2
        }
    }
    
    cfg = configs.get(config, configs['default'])
    cfg.update(kwargs)
    
    return DefectLoCKv2Improved(num_classes=num_classes, model_size=model_size, **cfg)


# ============================================================================
# 8. Wrapper for train_unified.py compatibility
# ============================================================================

class DefectLoCKv2ImprovedWrapper(nn.Module):
    """Wrapper for compatibility with train_unified.py"""
    
    def __init__(self, model: DefectLoCKv2Improved):
        super().__init__()
        self.model = model
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.model(x)


def get_defect_lock_v2_improved_for_training(
    num_classes: int = 7,
    model_size: str = 't',
    config: str = 'default',
    **kwargs
) -> DefectLoCKv2ImprovedWrapper:
    """Factory function for training"""
    model = get_defect_lock_v2_improved(num_classes, model_size, config=config, **kwargs)
    return DefectLoCKv2ImprovedWrapper(model)


# ============================================================================
# 9. Aliases for SAAD-Net
# ============================================================================

SAADNet = DefectLoCKv2Improved
get_saad_net = get_defect_lock_v2_improved
get_saad_net_for_training = get_defect_lock_v2_improved_for_training
SAADNetLoss = DefectLoCKv2ImprovedLoss


# ============================================================================
# 10. Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing DefectLoCK V2 Improved (SAAD-Net)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"TIMM available: {TIMM_AVAILABLE}")
    print(f"Torchvision Swin V2 available: {SWIN_V2_AVAILABLE}")
    
    # Test with 1024x1024 (timm backend)
    print("\n" + "=" * 50)
    print("Test 1: img_size=1024 (timm backend)")
    print("=" * 50)
    
    model = get_defect_lock_v2_improved(
        num_classes=7, model_size='t', config='default', img_size=1024
    )
    model = model.to(device)
    
    print(f"Backend: {model.backend}")
    print(f"Feature dim: {model.feat_dim}")
    
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total:.2f}M")
    
    x = torch.randn(2, 3, 1024, 1024).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Logits: {outputs['logits'].shape}")
    print(f"Features: {outputs['features'].shape}")
    
    # Test with 256x256 (torchvision backend)
    print("\n" + "=" * 50)
    print("Test 2: img_size=256 (torchvision backend)")
    print("=" * 50)
    
    model2 = get_defect_lock_v2_improved(
        num_classes=7, model_size='t', config='default', img_size=256
    )
    model2 = model2.to(device)
    
    print(f"Backend: {model2.backend}")
    
    x2 = torch.randn(2, 3, 256, 256).to(device)
    model2.eval()
    with torch.no_grad():
        outputs2 = model2(x2)
    
    print(f"Input: {x2.shape}")
    print(f"Logits: {outputs2['logits'].shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)