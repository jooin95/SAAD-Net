#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAAD-Net: Scale-Adaptive Anomaly Detection Network
====================================================
Three enhancements over baseline (cbam_laa):
  1. FPN    : Swin V2 Stage 1~4  feature -> P2~P5 (256ch)
  2. PCA    : Patch-level Contrastive Alignment ( CDL ->    )
  3. AFHEAD : Anchor-free Detection Head (FCOS-style, LAA anomaly map centerness )

 :
  Input
    +-> Swin V2 Backbone (4-stage feature )
         +-> FPN (P2~P5, 256ch )
              +-> CBAM ( FPN )
                   +-> LAA (anomaly map , centerness )
                        +-> Anchor-free Detection Head (cls / bbox / centerness)
                        +-> Patch Contrastive Alignment Head

  :
  - Classification label   (weakly supervised):  
  - Bounding Box annotation  :   (bbox_targets  )

Paper Ablation:
  - baseline        : Swin V2-T only
  - cbam_laa        : + CBAM + LAA  ( best)
  - saad_fpn        : + FPN
  - saad_fpn_pca    : + FPN + PCA
  - saad_default    : Full (FPN + CBAM + LAA + PCA + AFHEAD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
import math

try:
    from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
    from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
    SWIN_V2_AVAILABLE = True
except ImportError:
    SWIN_V2_AVAILABLE = False


# ============================================================================
# 0. Swin V2  Feature Extractor
# ============================================================================

# Swin V2-T stage  
_BACKBONE_CHANNELS = {
    't': [96, 192, 384, 768],
    's': [96, 192, 384, 768],
    'b': [128, 256, 512, 1024],
}

class SwinV2FeatureExtractor(nn.Module):
    """
    Swin V2  4 Stage  feature .

    torchvision Swin V2 :
      features[0] : PatchEmbed
      features[1] : Stage-1 (SwinTransformerBlock x2)  -> (B, H/4,  W/4,  96)
      features[2] : PatchMerging
      features[3] : Stage-2                             -> (B, H/8,  W/8,  192)
      features[4] : PatchMerging
      features[5] : Stage-3                             -> (B, H/16, W/16, 384)
      features[6] : PatchMerging
      features[7] : Stage-4                             -> (B, H/32, W/32, 768)

    : [(B,C1,H/4,W/4), (B,C2,H/8,W/8), (B,C3,H/16,W/16), (B,C4,H/32,W/32)]
    """

    # stage output indices in backbone.features
    _STAGE_INDICES = [1, 3, 5, 7]

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: Tensor) -> List[Tensor]:
        features = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self._STAGE_INDICES:
                # (B, H, W, C) -> (B, C, H, W)
                features.append(x.permute(0, 3, 1, 2).contiguous())
        return features  # [C1, C2, C3, C4]


# ============================================================================
# 1. FPN (Feature Pyramid Network)
# ============================================================================

class FPN(nn.Module):
    """
    Feature Pyramid Network

    Swin V2 Stage 1~4 -> P2~P5 (256ch)  .

    Top-down pathway + lateral connections:
      P5 <- C4 (lateral conv)
      P4 <- C3 (lateral conv) + upsample(P5)
      P3 <- C2 (lateral conv) + upsample(P4)
      P2 <- C1 (lateral conv) + upsample(P3)

    Shared output conv (3x3) applies to all levels.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1x1 convolutions ( )
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for c in in_channels
        ])

        # Output 3x3 convolutions ( FPN )
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels
        ])

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: [C1, C2, C3, C4] from coarse to fine
        Returns:
            [P2, P3, P4, P5] (fine to coarse, all 256ch)
        """
        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down merge (C4->C3->C2->C1)
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode='nearest'
            )

        # Output convolutions
        pyramids = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return pyramids  # [P2, P3, P4, P5]


# ============================================================================
# 2. CBAM
# ============================================================================

class ECABlock(nn.Module):
    """Efficient Channel Attention"""

    def __init__(self, channels: int):
        super().__init__()
        k = max(3, int(abs((math.log2(channels) + 1) / 2)) | 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
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
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ECABlock(channels)
        self.spatial_att = SpatialAttention(7)

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial_att(self.channel_att(x))


# ============================================================================
# 3. LAA (Lightweight Anomaly-Aware Attention)
# ============================================================================

class LightweightAnomalyAttention(nn.Module):
    """
    Lightweight Anomaly-Aware Attention (LAA)

    F_laa = F_ch * (1 + alpha . A)
     A spatial anomaly map (B, 1, H, W)
     anomaly map Anchor-free head centerness  .
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)

        # Channel attention (GAP + GMP concat -> MLP)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial anomaly map: 1x1 -> 7x7 -> Sigmoid
        self.spatial_anomaly = nn.Sequential(
            nn.Conv2d(channels, r, 1, bias=False),
            nn.BatchNorm2d(r),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # Learnable enhancement scale ( 0.1 ->  )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        avg = self.avg_pool(x)
        mx  = self.max_pool(x)
        ch_att = self.channel_mlp(torch.cat([avg, mx], dim=1))
        x_ch = x * ch_att

        anomaly_map = self.spatial_anomaly(x_ch)          # (B, 1, H, W)
        enhanced    = x_ch * (1 + self.alpha * anomaly_map)
        return enhanced, anomaly_map


# ============================================================================
# 4. Anchor-free Detection Head  (FCOS-style)
# ============================================================================

class AnchorFreeHead(nn.Module):
    """
    FCOS-style Anchor-free Detection Head

     FPN  3 :
      cls_logits  : (B, num_classes, H, W)   
      bbox_pred   : (B, 4,           H, W)    (l, t, r, b , exp )
      centerness  : (B, 1,           H, W)   (LAA anomaly map )

    Weakly supervised :
      bbox annotation  LAA anomaly map centerness  .
      -> classification loss + centerness-anomaly  loss   .
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared convolutional tower (cls / reg )
        def _tower(n):
            layers = []
            for _ in range(n):
                layers += [
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True)
                ]
            return nn.Sequential(*layers)

        self.cls_tower = _tower(num_convs)
        self.reg_tower = _tower(num_convs)

        # Output heads
        self.cls_head = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, 4,           3, padding=1)
        self.ctr_head = nn.Conv2d(in_channels, 1,           3, padding=1)

        # Per-level learnable scales (standard FCOS: each level has own scale)
        # P2 predicts small distances (0-64px), P5 predicts large (256px+)
        # Single shared scale prevents per-level specialization
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(4)])

        self._init_weights()

    def _init_weights(self):
        # Focal Loss bias  (pi=0.01)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))
        for m in [self.reg_head, self.ctr_head]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        anomaly_map: Optional[Tensor] = None,
        level_idx: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x          : FPN feature  (B, C, H, W)
            anomaly_map: LAA anomaly  (B, 1, H, W)  centerness  (optional)
            level_idx  : FPN level index 0~3 (P2~P5), selects per-level scale
        Returns:
            cls_logits  : (B, num_classes, H, W)
            bbox_pred   : (B, 4, H, W)  exp(.) -> pixel distances
            centerness  : (B, 1, H, W)  raw logit (sigmoid applied in decode/loss)
        """
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)

        cls_logits = self.cls_head(cls_feat)

        # Per-level scale: each level specializes for its distance range
        # P2(l=0): small defects 0-64px, P5(l=3): large defects 256px+
        bbox_pred = torch.exp(self.scales[level_idx] * self.reg_head(reg_feat))

        # Centerness: raw logit (NOT sigmoid here - decode/loss applies sigmoid once)
        # BUG FIX: previous code applied sigmoid here AND decode applied it again
        # -> sigmoid(sigmoid(x)) compresses scores to [0.73, 0.88], killing low-confidence dets
        centerness = self.ctr_head(reg_feat)
        if anomaly_map is not None:
            h, w = centerness.shape[-2:]
            am = F.interpolate(anomaly_map, size=(h, w), mode='bilinear', align_corners=False)
            centerness = centerness + am

        # Return raw centerness logit (sigmoid applied in FCOSLoss and decode_fcos_predictions)
        return cls_logits, bbox_pred, centerness


# ============================================================================
# 5. Patch Contrastive Alignment (PCA)
# ============================================================================

class PatchContrastiveAlignment(nn.Module):
    """
    Patch-level Contrastive Alignment (PCA)

     CDL()   :
      1. FPN feature map NxN   
      2. Patch projection head 
      3. SupCon loss:     ,    

    anomaly_map  :
      defect (anomaly_map > thresh)      
      ->    

    Args:
        in_channels   : FPN feature   (256)
        proj_dim      :    (128)
        patch_size    :    (7 -> 7x7=49 patches per image)
        temperature   : SupCon temperature
    """

    def __init__(
        self,
        in_channels: int = 256,
        proj_dim: int = 128,
        patch_size: int = 7,
        temperature: float = 0.07
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temperature = temperature

        # Patch-level projection ( -> MLP)
        self.patch_pool  = nn.AdaptiveAvgPool2d(patch_size)
        self.projector   = nn.Sequential(
            nn.Flatten(start_dim=2),               # (B, C, P)
            # Transpose -> (B, P, C)   
        )
        # Per-patch MLP (shared weights)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )

    def _extract_patch_embeddings(
        self,
        feat: Tensor,
        anomaly_map: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Returns:
            embeddings : (B, P, proj_dim)  L2 normalized
            patch_weights: (B, P)  anomaly  (optional)
        """
        B, C, H, W = feat.shape
        P = self.patch_size

        # (B, C, P, P) -> (B, C, P) -> (B, P, C)
        pooled = self.patch_pool(feat)                       # (B, C, P, P)
        patches = pooled.flatten(2).permute(0, 2, 1)        # (B, P, C)

        # Flatten batch x patch for BN
        flat = patches.reshape(B * P * P, C)
        flat = self.mlp(flat)                                # (B*P, proj_dim)
        embs = flat.reshape(B, P * P, -1)
        embs = F.normalize(embs, dim=-1)

        # Anomaly-aware   (  )
        weights = None
        if anomaly_map is not None:
            am = F.adaptive_avg_pool2d(anomaly_map, (P, P))  # (B, 1, P, P)
            weights = am.flatten(1).detach()                  # (B, P)
            weights = (weights - weights.min(1, keepdim=True).values) / \
                      (weights.max(1, keepdim=True).values - weights.min(1, keepdim=True).values + 1e-8)

        return embs, weights

    def forward(
        self,
        feat: Tensor,
        labels: Tensor,
        anomaly_map: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            feat       : FPN feature (B, C, H, W)
            labels     : class labels (B,)
            anomaly_map: LAA anomaly (B, 1, H, W) [optional]
        Returns:
            patch_contrastive_loss: scalar
        """
        B = feat.size(0)
        if B < 2:
            return torch.tensor(0.0, device=feat.device)

        embs, weights = self._extract_patch_embeddings(feat, anomaly_map)
        # embs: (B, P, dim)

        P2 = embs.size(1)
        loss_total = torch.tensor(0.0, device=feat.device)
        count = 0

        #      contrastive 
        for p in range(P2):
            z = embs[:, p, :]                                # (B, dim)
            sim = torch.matmul(z, z.T) / self.temperature   # (B, B)

            # Positive mask ( ,   )
            label_mat  = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask   = label_mat.float() - torch.eye(B, device=feat.device)
            pos_mask   = pos_mask.clamp(min=0)

            if pos_mask.sum() == 0:
                continue

            # SupCon loss
            logits_max, _ = sim.max(1, keepdim=True)
            logits = sim - logits_max.detach()

            exp_logits = torch.exp(logits)
            no_diag    = 1 - torch.eye(B, device=feat.device)
            log_prob   = logits - torch.log((exp_logits * no_diag).sum(1, keepdim=True) + 1e-8)

            n_pos         = pos_mask.sum(1).clamp(min=1)
            mean_log_prob = (pos_mask * log_prob).sum(1) / n_pos

            # Anomaly   (    )
            if weights is not None:
                w = weights[:, p]
                loss_p = -(w * mean_log_prob).mean()
            else:
                loss_p = -mean_log_prob.mean()

            loss_total = loss_total + loss_p
            count += 1

        return loss_total / max(count, 1)


# ============================================================================
# 6. Main Model: SAAD-Net
# ============================================================================

class SAADNet(nn.Module):
    """
    SAAD-Net: Scale-Adaptive Anomaly Detection Network

    Architecture:
      Input
        +-> Swin V2 Feature Extractor (4 stages)
             +-> FPN (P2~P5, fpn_channels)
                  +-> CBAM (per FPN level, optional)
                       +-> LAA  (P3   anomaly map , optional)
                            +-> Anchor-free Detection Head (per FPN level, optional)
                            +-> Patch Contrastive Alignment (P3 , optional)

    P   :
      - Detection head:   (P2~P5)
      - LAA + PCA:      P3 (stride=8,    )
      - Classification: P5 GAP ( )

    Args:
        num_classes          :   
        model_size           : 't' | 's' | 'b'
        pretrained           : ImageNet-22K pretrained  
        fpn_channels         : FPN   (default: 256)
        use_cbam             : CBAM 
        use_laa              : LAA 
        use_detection_head   : Anchor-free head 
        use_patch_contrastive: PCA 
        dropout              : Classifier dropout
    """

    def __init__(
        self,
        num_classes: int = 7,
        model_size: str = 't',
        pretrained: bool = True,
        fpn_channels: int = 256,
        use_cbam: bool = True,
        use_laa: bool = True,
        use_detection_head: bool = True,
        use_patch_contrastive: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.use_cbam              = use_cbam
        self.use_laa               = use_laa
        self.use_detection_head    = use_detection_head
        self.use_patch_contrastive = use_patch_contrastive
        self.fpn_channels          = fpn_channels
        self.num_classes           = num_classes

        # ==================== Backbone ====================
        if model_size == 't':
            weights        = Swin_V2_T_Weights.DEFAULT if pretrained else None
            backbone       = swin_v2_t(weights=weights)
            backbone_ch    = _BACKBONE_CHANNELS['t']
        elif model_size == 's':
            weights        = Swin_V2_S_Weights.DEFAULT if pretrained else None
            backbone       = swin_v2_s(weights=weights)
            backbone_ch    = _BACKBONE_CHANNELS['s']
        elif model_size == 'b':
            weights        = Swin_V2_B_Weights.DEFAULT if pretrained else None
            backbone       = swin_v2_b(weights=weights)
            backbone_ch    = _BACKBONE_CHANNELS['b']
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        backbone.head = nn.Identity()   #   head 
        self.feature_extractor = SwinV2FeatureExtractor(backbone)

        # ==================== FPN ====================
        self.fpn = FPN(backbone_ch, fpn_channels)

        # ==================== CBAM (per FPN level) ====================
        if use_cbam:
            self.cbam_modules = nn.ModuleList([
                CBAM(fpn_channels) for _ in range(4)
            ])

        # ==================== LAA (P3 ) ====================
        if use_laa:
            self.laa = LightweightAnomalyAttention(fpn_channels)

        # ==================== Anchor-free Detection Head ====================
        if use_detection_head:
            self.det_head = AnchorFreeHead(fpn_channels, num_classes, num_convs=4)
            # FPN strides: P2=4, P3=8, P4=16, P5=32
            self.register_buffer(
                'fpn_strides',
                torch.tensor([4, 8, 16, 32], dtype=torch.float32)
            )

        # ==================== Patch Contrastive Alignment ====================
        if use_patch_contrastive:
            self.pca = PatchContrastiveAlignment(
                in_channels=fpn_channels,
                proj_dim=128,
                patch_size=7,
                temperature=0.07
            )

        # ==================== Classification Head ====================
        # P5 (stride=32) GAP -> Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(fpn_channels),
            nn.Dropout(dropout),
            nn.Linear(fpn_channels, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None
    ) -> Dict[str, object]:
        """
        Args:
            x      : (B, 3, H, W)
            labels : (B,)  PCA   , inference   

        Returns dict:
            'logits'       : (B, num_classes)           classification
            'features'     : (B, fpn_channels)           global feature (P5 GAP)
            'pyramids'     : list of (B, 256, Hi, Wi)    FPN features
            'anomaly_map'  : (B, 1, H_p3, W_p3)         LAA anomaly (if use_laa)
            'det_outputs'  : list of (cls, bbox, ctr)    per FPN level (if use_detection_head)
            'pca_loss'     : scalar                      patch contrastive (if use_patch_contrastive & training)
        """
        # -- 1. Backbone multi-scale features ------------------------------
        backbone_features = self.feature_extractor(x)   # [C1,C2,C3,C4]

        # -- 2. FPN -> [P2, P3, P4, P5] ------------------------------------
        pyramids = self.fpn(backbone_features)           # 4 x (B,256,Hi,Wi)

        # -- 3. CBAM ( FPN ) -----------------------------------------
        if self.use_cbam:
            pyramids = [cbam(p) for cbam, p in zip(self.cbam_modules, pyramids)]

        # -- 4. LAA (P3 = index 1, stride=8  ) -------------------
        anomaly_map = None
        if self.use_laa:
            pyramids[1], anomaly_map = self.laa(pyramids[1])

        # -- 5. Anchor-free Detection Head ( FPN ) -----------------
        det_outputs = None
        if self.use_detection_head:
            det_outputs = []
            for i, p in enumerate(pyramids):
                # P3(i=1) anomaly_map  -> centerness 
                am = anomaly_map if (i == 1 and anomaly_map is not None) else None
                cls_l, bbox_p, ctr = self.det_head(p, am, level_idx=i)
                det_outputs.append({
                    'cls_logits': cls_l,
                    'bbox_pred' : bbox_p,
                    'centerness': ctr,
                    'stride'    : float(self.fpn_strides[i])
                })

        # -- 6. Classification (P5 GAP) ------------------------------------
        p5       = pyramids[-1]                          # (B, 256, H/32, W/32)
        features = self.classifier[0](p5).flatten(1)    # GAP -> (B, 256)
        logits   = self.classifier[1:](
            self.classifier[0](p5).flatten(1)
        )
        # re-compute cleanly
        logits   = self.classifier(p5)                  # (B, num_classes)
        with torch.no_grad():
            features = F.adaptive_avg_pool2d(p5, 1).flatten(1)

        # -- 7. Patch Contrastive Alignment (  + labels  ) -----
        pca_loss = None
        if self.use_patch_contrastive and labels is not None and self.training:
            pca_loss = self.pca(pyramids[1], labels, anomaly_map)

        # --   --------------------------------------------------
        outputs = {
            'logits'     : logits,
            'features'   : features,
            'pyramids'   : pyramids,
            'img_size'   : (x.shape[2], x.shape[3]),   # (H, W)  FCOS target assignment
        }
        if anomaly_map is not None:
            outputs['anomaly_map'] = anomaly_map
        if det_outputs is not None:
            outputs['det_outputs'] = det_outputs
        if pca_loss is not None:
            outputs['pca_loss'] = pca_loss

        return outputs


# ============================================================================
# 7. Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing"""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce  = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt  = torch.exp(-F.cross_entropy(logits, targets, reduction='none'))
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================================
# 7-A. FCOS Target Assigner (GIoU  supervised)
# ============================================================================

def _box_giou_loss(pred_ltrb: Tensor, gt_ltrb: Tensor) -> Tensor:
    """
    GIoU Loss for FCOS box representation (l, t, r, b distances from pixel center).

    Args:
        pred_ltrb : (N, 4)   [left, top, right, bottom] distances
        gt_ltrb   : (N, 4)  GT    [left, top, right, bottom] distances
    Returns:
        giou_loss : (N,)  1 - GIoU
    """
    #  
    pred_w = (pred_ltrb[:, 0] + pred_ltrb[:, 2]).clamp(min=1e-6)
    pred_h = (pred_ltrb[:, 1] + pred_ltrb[:, 3]).clamp(min=1e-6)
    gt_w   = (gt_ltrb[:, 0]   + gt_ltrb[:, 2]).clamp(min=1e-6)
    gt_h   = (gt_ltrb[:, 1]   + gt_ltrb[:, 3]).clamp(min=1e-6)

    pred_area = pred_w * pred_h
    gt_area   = gt_w   * gt_h

    # Intersection
    inter_w = (torch.min(pred_ltrb[:, 0], gt_ltrb[:, 0])
               + torch.min(pred_ltrb[:, 2], gt_ltrb[:, 2])).clamp(min=0)
    inter_h = (torch.min(pred_ltrb[:, 1], gt_ltrb[:, 1])
               + torch.min(pred_ltrb[:, 3], gt_ltrb[:, 3])).clamp(min=0)
    inter   = inter_w * inter_h

    union = pred_area + gt_area - inter + 1e-6
    iou   = inter / union

    # Enclosing box
    enc_w = (torch.max(pred_ltrb[:, 0], gt_ltrb[:, 0])
             + torch.max(pred_ltrb[:, 2], gt_ltrb[:, 2])).clamp(min=1e-6)
    enc_h = (torch.max(pred_ltrb[:, 1], gt_ltrb[:, 1])
             + torch.max(pred_ltrb[:, 3], gt_ltrb[:, 3])).clamp(min=1e-6)
    enc   = enc_w * enc_h

    giou = iou - (enc - union) / (enc + 1e-6)
    return 1.0 - giou


def _fcos_assign_level(
    boxes_norm: Tensor,          # (N, 5) [cls, cx_n, cy_n, w_n, h_n]
    feat_h: int,
    feat_w: int,
    stride: int,
    img_h: int,
    img_w: int,
    size_min: float,
    size_max: float,
    num_classes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
     FPN  FCOS target  ( ).

    Returns:
        cls_t  : (Hf, Wf) int64   0=bg, 1~C=fg class, -1=ignore
        reg_t  : (Hf, Wf, 4)     l, t, r, b (pixel )
        ctr_t  : (Hf, Wf)        centerness target [0,1]
        fg_mask: (Hf, Wf) bool
    """
    device = boxes_norm.device
    cls_t  = torch.zeros(feat_h, feat_w, dtype=torch.long,  device=device)
    reg_t  = torch.zeros(feat_h, feat_w, 4, dtype=torch.float32, device=device)
    ctr_t  = torch.zeros(feat_h, feat_w, dtype=torch.float32, device=device)
    fg_mask = torch.zeros(feat_h, feat_w, dtype=torch.bool, device=device)

    if boxes_norm.shape[0] == 0:
        return cls_t, reg_t, ctr_t, fg_mask

    #    (img )
    ys = (torch.arange(feat_h, device=device).float() + 0.5) * stride  # (Hf,)
    xs = (torch.arange(feat_w, device=device).float() + 0.5) * stride  # (Wf,)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')             # (Hf, Wf)

    # GT boxes -> absolute pixel coords
    cls_ids = boxes_norm[:, 0].long()                      # (N,)
    cx = boxes_norm[:, 1] * img_w                          # (N,)
    cy = boxes_norm[:, 2] * img_h
    bw = boxes_norm[:, 3] * img_w
    bh = boxes_norm[:, 4] * img_h
    x1 = cx - bw / 2;  x2 = cx + bw / 2
    y1 = cy - bh / 2;  y2 = cy + bh / 2

    # area (    )
    areas = bw * bh                                        # (N,)

    #    (     )
    order = torch.argsort(areas, descending=True)
    cls_ids = cls_ids[order]
    x1 = x1[order]; y1 = y1[order]; x2 = x2[order]; y2 = y2[order]
    areas = areas[order]

    for n in range(len(order)):
        #  GT box 
        in_box = (
            (grid_x >= x1[n]) & (grid_x <= x2[n]) &
            (grid_y >= y1[n]) & (grid_y <= y2[n])
        )  # (Hf, Wf) bool

        if not in_box.any():
            continue

        # l, t, r, b distances
        l = (grid_x - x1[n]).clamp(min=0)
        t = (grid_y - y1[n]).clamp(min=0)
        r = (x2[n]  - grid_x).clamp(min=0)
        b = (y2[n]  - grid_y).clamp(min=0)
        max_dist = torch.stack([l, t, r, b], dim=-1).max(dim=-1).values  # (Hf,Wf)

        #       (FCOS : max(l,t,r,b) )
        in_range = (max_dist >= size_min) & (max_dist <= size_max)
        valid = in_box & in_range

        if not valid.any():
            continue

        # centerness: sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
        lr_min = torch.min(l, r).clamp(min=0)
        lr_max = torch.max(l, r).clamp(min=1e-6)
        tb_min = torch.min(t, b).clamp(min=0)
        tb_max = torch.max(t, b).clamp(min=1e-6)
        ctr    = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))

        fg_mask[valid]            = True
        cls_t[valid]              = cls_ids[n] + 1                    # 1-indexed fg
        reg_t[valid]              = torch.stack([l, t, r, b], dim=-1)[valid]
        ctr_t[valid]              = ctr[valid]

    return cls_t, reg_t, ctr_t, fg_mask


class FCOSLoss(nn.Module):
    """
    FCOS-style Detection Loss   supervised (GIoU + Focal + Centerness)

    bbox annotation  fully supervised,
     (boxes_list=None    ) weakly supervised fallback.

    Fully supervised:
      cls  : sigmoid focal loss per foreground pixel
      bbox : GIoU loss per foreground pixel
      ctr  : BCE per foreground pixel

    Weakly supervised fallback:
      cls  : global max-pool -> cross-entropy (image-level)
      ctr  : defect  BCE
    """

    # FCOS   size range per FPN level (P2~P5)
    LEVEL_SIZE_RANGES = [
        (0,    64),     # P2  (stride=4)
        (64,   128),    # P3  (stride=8)
        (128,  256),    # P4  (stride=16)
        (256,  1e8),    # P5  (stride=32)
    ]
    FPN_STRIDES = [4, 8, 16, 32]

    def __init__(self, num_classes: int, focal_gamma: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def _sigmoid_focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Sigmoid Focal Loss (per element).
        logits : (N, C)  raw logits
        targets: (N,)    int64  1~C = fg class, 0 = bg  -> one-hot
        """
        C = logits.shape[1]
        one_hot = F.one_hot(targets.clamp(min=0), C + 1)[:, 1:].float()  # skip bg col
        p  = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, one_hot, reduction='none')
        pt = p * one_hot + (1 - p) * (1 - one_hot)
        w  = self.focal_alpha * one_hot + (1 - self.focal_alpha) * (1 - one_hot)
        return (w * (1 - pt) ** self.focal_gamma * ce).sum(dim=-1)

    def forward(
        self,
        det_outputs  : List[Dict],
        targets      : Tensor,                        # (B,) image-level class
        img_size     : Tuple[int, int],               # (H, W) original image
        boxes_list   : Optional[List[Tensor]] = None, # List[Tensor(N,5)]
    ) -> Dict[str, Tensor]:

        device   = targets.device
        B        = targets.shape[0]
        img_h, img_w = img_size

        # bbox  
        has_bbox = (
            boxes_list is not None and
            any(b.shape[0] > 0 for b in boxes_list)
        )

        # -- Fully supervised ----------------------------------------------
        if has_bbox:
            total_cls  = torch.tensor(0.0, device=device)
            total_bbox = torch.tensor(0.0, device=device)
            total_ctr  = torch.tensor(0.0, device=device)
            total_fg   = 0

            for lvl_idx, out in enumerate(det_outputs):
                cls_l = out['cls_logits']  # (B, C, Hf, Wf)
                bbox_p = out['bbox_pred']  # (B, 4, Hf, Wf)  exp activated
                ctr_p  = out['centerness'] # (B, 1, Hf, Wf)

                B_, C_, Hf, Wf = cls_l.shape
                stride   = self.FPN_STRIDES[lvl_idx]
                sz_min, sz_max = self.LEVEL_SIZE_RANGES[lvl_idx]

                # --  target  --------------------------------
                for b in range(B_):
                    boxes_b = boxes_list[b]                    # (N, 5)

                    cls_t, reg_t, ctr_t, fg = _fcos_assign_level(
                        boxes_norm = boxes_b,
                        feat_h=Hf, feat_w=Wf,
                        stride=stride,
                        img_h=img_h, img_w=img_w,
                        size_min=sz_min, size_max=sz_max,
                        num_classes=self.num_classes,
                    )
                    # cls_t  : (Hf, Wf)  0=bg, 1~C=fg
                    # reg_t  : (Hf, Wf, 4)
                    # ctr_t  : (Hf, Wf)
                    # fg     : (Hf, Wf) bool

                    n_fg = fg.sum().item()
                    if n_fg == 0:
                        continue
                    total_fg += n_fg

                    # Foreground pixel 
                    cls_pred_fg  = cls_l[b].permute(1, 2, 0)[fg]   # (n_fg, C)
                    bbox_pred_fg = bbox_p[b].permute(1, 2, 0)[fg]  # (n_fg, 4)
                    ctr_pred_fg  = torch.sigmoid(ctr_p[b, 0][fg])  # (n_fg,) sigmoid here

                    cls_tgt_fg   = cls_t[fg]                         # (n_fg,) 1~C
                    reg_tgt_fg   = reg_t[fg]                         # (n_fg, 4)
                    ctr_tgt_fg   = ctr_t[fg]                         # (n_fg,)

                    # 1. Cls: Sigmoid Focal Loss
                    total_cls = total_cls + self._sigmoid_focal_loss(
                        cls_pred_fg, cls_tgt_fg
                    ).sum()

                    # 2. BBox: GIoU Loss (centerness weighting )
                    giou = _box_giou_loss(bbox_pred_fg, reg_tgt_fg)  # (n_fg,)
                    total_bbox = total_bbox + (giou * ctr_tgt_fg).sum()

                    # 3. Centerness: BCE with logits (head returns raw logit now)
                    with torch.cuda.amp.autocast(enabled=False):
                        total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
                            ctr_pred_fg.float(), ctr_tgt_fg.float(), reduction='sum'
                        )

            norm = max(total_fg, 1)
            return {
                'cls' : total_cls  / norm,
                'bbox': total_bbox / norm,
                'ctr' : total_ctr  / norm,
            }

        # -- Weakly supervised fallback ------------------------------------
        else:
            cls_loss_total = torch.tensor(0.0, device=device)
            ctr_loss_total = torch.tensor(0.0, device=device)

            for out in det_outputs:
                cls_l = out['cls_logits']    # (B, C, H, W)
                ctr   = out['centerness']    # (B, 1, H, W) raw logit

                cls_pooled     = cls_l.amax(dim=(-2, -1))              # (B, C)
                cls_loss_total = cls_loss_total + F.cross_entropy(cls_pooled, targets)

                is_defect  = (targets > 0).float().view(-1, 1, 1, 1)
                ctr_target = is_defect.expand_as(ctr)
                with torch.cuda.amp.autocast(enabled=False):
                    # ctr is raw logit now -> use binary_cross_entropy_with_logits
                    ctr_loss_total = ctr_loss_total + F.binary_cross_entropy_with_logits(
                        ctr.float(), ctr_target.float()
                    )

            n = len(det_outputs)
            return {
                'cls' : cls_loss_total / n,
                'bbox': torch.tensor(0.0, device=device),
                'ctr' : ctr_loss_total / n,
            }


class SAADNetLoss(nn.Module):
    """
    SAAD-Net  Loss

    Total = lambda_focal x Focal
          + lambda_det   x (FCOS_cls + FCOS_bbox + 0.5xFCOS_ctr)
          + lambda_t     x PCA  (progressive warmup)

    Fully supervised:  FCOS_bbox = GIoU loss (per foreground pixel)
    Weakly supervised: FCOS_bbox = 0  (bbox annotation    fallback)

    lambda_t = lambda_pca x min(t / T_warmup, 1.0)   progressive warmup
    """

    def __init__(
        self,
        num_classes:     int   = 7,
        focal_gamma:     float = 2.0,
        label_smoothing: float = 0.1,
        lambda_det:      float = 0.5,
        lambda_pca:      float = 0.3,
        temperature:     float = 0.07,
    ):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_pca = lambda_pca

        self.focal_loss = FocalLoss(focal_gamma, label_smoothing)
        self.fcos_loss  = FCOSLoss(num_classes, focal_gamma=focal_gamma)

    def forward(
        self,
        outputs    : Dict,
        targets    : Tensor,                      # (B,) image-level class labels
        boxes_list : Optional[List[Tensor]] = None, # List[Tensor(N,5)] VOC bbox (normalized)
        pca_weight : Optional[float] = None,      # progressive warmup override
    ) -> Dict[str, Tensor]:
        """
        Args:
            outputs    : SAADNet forward  
            targets    : (B,) class labels
            boxes_list : List of Tensor(N,5) [cls_idx, cx_n, cy_n, w_n, h_n]
                         None     -> weakly supervised fallback
            pca_weight : CDL progressive warmup weight override
        """
        device = targets.device

        # 1. Focal Loss (classification, P5 logits)
        loss_focal = self.focal_loss(outputs['logits'], targets)

        # 2. Detection Loss (FCOS fully/weakly supervised)
        loss_det_cls  = torch.tensor(0.0, device=device)
        loss_det_bbox = torch.tensor(0.0, device=device)
        loss_det_ctr  = torch.tensor(0.0, device=device)

        if 'det_outputs' in outputs and outputs['det_outputs']:
            img_size = outputs.get('img_size', (512, 512))
            det_losses = self.fcos_loss(
                det_outputs = outputs['det_outputs'],
                targets     = targets,
                img_size    = img_size,
                boxes_list  = boxes_list,
            )
            loss_det_cls  = det_losses['cls']
            loss_det_bbox = det_losses['bbox']
            loss_det_ctr  = det_losses['ctr']

        # 3. Patch Contrastive Alignment
        loss_pca = outputs.get('pca_loss', torch.tensor(0.0, device=device))
        if not isinstance(loss_pca, Tensor):
            loss_pca = torch.tensor(0.0, device=device)

        w_pca = pca_weight if pca_weight is not None else self.lambda_pca

        total = (loss_focal
                 + self.lambda_det * (loss_det_cls + loss_det_bbox + 0.5 * loss_det_ctr)
                 + w_pca * loss_pca)

        return {
            'total'    : total,
            'focal'    : loss_focal,
            'det_cls'  : loss_det_cls,
            'det_bbox' : loss_det_bbox,
            'det_ctr'  : loss_det_ctr,
            'pca'      : loss_pca,
        }


# ============================================================================
# 8. Factory Functions
# ============================================================================

def get_saad_net(
    num_classes: int  = 7,
    model_size:  str  = 't',
    config:      str  = 'saad_default',
    **kwargs
) -> SAADNet:
    """
    SAAD-Net Factory Function

    Ablation Configs:
    -------------------------------------------------
    baseline       : Swin V2-T only (FPN ,  )
    cbam_laa       : + CBAM + LAA  ( best )
    saad_fpn       : + FPN  (FPN )
    saad_fpn_pca   : + FPN + PCA
    saad_default   : Full (FPN + CBAM + LAA + PCA + AFHEAD)
    -------------------------------------------------

    Usage:
        model = get_saad_net(num_classes=7, config='saad_default')
        loss  = SAADNetLoss(num_classes=7)

        outputs = model(images, labels)
        losses  = loss(outputs, labels, pca_weight=current_pca_weight)
    """
    configs = {
        # -- Ablation -------------------------------------------------------
        'baseline': dict(
            use_cbam=False, use_laa=False,
            use_detection_head=False, use_patch_contrastive=False,
        ),
        'cbam_laa': dict(
            use_cbam=True,  use_laa=True,
            use_detection_head=False, use_patch_contrastive=False,
        ),
        'saad_fpn': dict(
            use_cbam=True,  use_laa=True,
            use_detection_head=False, use_patch_contrastive=False,
        ),
        'saad_fpn_pca': dict(
            use_cbam=True,  use_laa=True,
            use_detection_head=False, use_patch_contrastive=True,
        ),
        # -- Main ----------------------------------------------------------
        'saad_default': dict(
            use_cbam=True,  use_laa=True,
            use_detection_head=True,  use_patch_contrastive=True,
        ),
        'saad_light': dict(
            use_cbam=True,  use_laa=True,
            use_detection_head=True,  use_patch_contrastive=False,
        ),
    }

    cfg = configs.get(config, configs['saad_default'])
    cfg.update(kwargs)

    return SAADNet(
        num_classes=num_classes,
        model_size=model_size,
        **cfg
    )


def get_saad_net_loss(
    num_classes:     int   = 7,
    lambda_det:      float = 0.5,
    lambda_pca:      float = 0.3,
    focal_gamma:     float = 2.0,
    label_smoothing: float = 0.1,
) -> SAADNetLoss:
    """Loss factory"""
    return SAADNetLoss(
        num_classes=num_classes,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        lambda_det=lambda_det,
        lambda_pca=lambda_pca,
    )


# ============================================================================
# 9. train_unified.py  Wrapper
# ============================================================================

class SAADNetWrapper(nn.Module):
    """train_unified.py DefectLoCK    """

    def __init__(self, model: SAADNet):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, labels: Optional[Tensor] = None) -> Dict:
        return self.model(x, labels)


def get_saad_net_for_training(
    num_classes: int = 7,
    model_size:  str = 't',
    config:      str = 'saad_default',
    **kwargs
) -> SAADNetWrapper:
    model = get_saad_net(num_classes, model_size, config=config, **kwargs)
    return SAADNetWrapper(model)


# ============================================================================
# 10. Quick Test
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SAAD-Net  Ablation Config Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    ablation_configs = [
        'baseline',
        'cbam_laa',
        'saad_fpn',
        'saad_fpn_pca',
        'saad_default',
    ]

    for cfg_name in ablation_configs:
        print(f"{'-'*55}")
        print(f"Config : {cfg_name}")

        model  = get_saad_net(num_classes=7, model_size='t', config=cfg_name)
        model  = model.to(device).train()
        loss_fn = get_saad_net_loss(num_classes=7)

        total_p     = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Params : {total_p:.2f}M total | {trainable_p:.2f}M trainable")

        x      = torch.randn(4, 3, 512, 512).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(x, labels)

        losses = loss_fn(outputs, targets=labels, boxes_list=None)

        print(f"Logits     : {outputs['logits'].shape}")
        print(f"FPN levels : {len(outputs['pyramids'])}  "
              f"shapes={[tuple(p.shape[-2:]) for p in outputs['pyramids']]}")
        if 'anomaly_map' in outputs:
            print(f"Anomaly map: {outputs['anomaly_map'].shape}")
        if 'det_outputs' in outputs:
            d = outputs['det_outputs']
            print(f"Det levels : {len(d)}  cls={d[0]['cls_logits'].shape}")
        if 'pca_loss' in outputs:
            print(f"PCA loss   : {outputs['pca_loss'].item():.4f}")

        has_bbox_str = 'weakly' if 'det_bbox' not in losses or losses['det_bbox'].item()==0 else 'fully'
        print(f"Total loss : {losses['total'].item():.4f}  "
              f"focal={losses['focal'].item():.4f}  "
              f"det_cls={losses['det_cls'].item():.4f}  "
              f"det_bbox={losses['det_bbox'].item():.4f}({has_bbox_str})  "
              f"pca={losses['pca'].item():.4f}")

    print("\n" + "=" * 70)
    print("All configs passed!")
    print("=" * 70)