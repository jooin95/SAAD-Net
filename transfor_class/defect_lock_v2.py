# -*- coding: utf-8 -*-
"""
DefectLoCK V2: Pretrained Swin V2 Backbone + Defect-specific Innovations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict
import math

from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class AnomalyScoreGenerator(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            LayerNorm2d(hidden_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            LayerNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1)
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            LayerNorm2d(channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        deviation = torch.abs(x - recon)
        anomaly_score = self.score_head(deviation)
        return anomaly_score, recon


class ScaleAdaptiveAttention(nn.Module):
    def __init__(self, channels: int, num_scales: int = 4):
        super().__init__()
        self.num_scales = num_scales
        self.branch_channels = channels // num_scales
        
        self.local_branches = nn.ModuleList()
        kernel_sizes = [3, 5, 7, 11]
        
        for i, k in enumerate(kernel_sizes[:num_scales]):
            branch = nn.Sequential(
                nn.Conv2d(channels, self.branch_channels, 1),
                nn.Conv2d(self.branch_channels, self.branch_channels, 
                         k, padding=k // 2, groups=self.branch_channels),
                LayerNorm2d(self.branch_channels),
                nn.GELU()
            )
            self.local_branches.append(branch)
        
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.branch_channels, 1),
            nn.GELU()
        )
        
        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, num_scales + 1),
            nn.Softmax(dim=-1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.branch_channels, channels, 1),
            LayerNorm2d(channels),
            nn.GELU()
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, C, H, W = x.shape
        
        local_feats = []
        for branch in self.local_branches:
            local_feats.append(branch(x))
        
        scale_weights = self.scale_predictor(x)
        
        local_combined = torch.zeros_like(local_feats[0])
        for i, feat in enumerate(local_feats):
            weight = scale_weights[:, i:i+1, None, None]
            local_combined = local_combined + weight * feat
        
        global_feat = self.global_branch(x)
        global_weight = scale_weights[:, -1:, None, None]
        global_feat = global_feat.expand(-1, -1, H, W) * global_weight
        
        combined = torch.cat([local_combined, global_feat], dim=1)
        fused = self.fusion(combined)
        attn = self.attention(fused)
        output = x + self.gamma * (x * attn)
        
        return output, scale_weights


class ContrastiveDefectHead(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, features: Tensor) -> Tensor:
        embeddings = self.projector(features)
        embeddings = self.norm(embeddings)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        if batch_size < 4:
            return torch.tensor(0.0, device=device)
        
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        
        loss = -mean_log_prob_pos.mean()
        return loss


class DefectLoCKv2(nn.Module):
    """
    DefectLoCK V2: Pretrained Swin V2 + Defect-specific Modules
    
    Architecture:
    - Backbone: Pretrained Swin V2 (frozen or fine-tuned)
    - Anomaly Module: Generates anomaly score map
    - SAA: Scale-Adaptive Attention for multi-scale defects
    - CDL: Contrastive learning for feature discrimination
    """
    def __init__(
        self,
        num_classes: int = 7,
        model_size: str = 't',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_saa: bool = True,
        use_anomaly: bool = True,
        use_contrastive: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_saa = use_saa
        self.use_anomaly = use_anomaly
        self.use_contrastive = use_contrastive
        
        # Load pretrained Swin V2 backbone
        if model_size == 't':
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            backbone = swin_v2_t(weights=weights)
            self.feat_dim = 768
        elif model_size == 's':
            weights = Swin_V2_S_Weights.DEFAULT if pretrained else None
            backbone = swin_v2_s(weights=weights)
            self.feat_dim = 768
        else:  # 'b'
            weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
            backbone = swin_v2_b(weights=weights)
            self.feat_dim = 1024
        
        # Extract backbone features (remove classification head)
        self.patch_embed = backbone.features[0]
        self.stages = nn.ModuleList([
            backbone.features[i] for i in range(1, len(backbone.features))
        ])
        self.norm = backbone.norm
        self.permute = backbone.permute
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for stage in self.stages:
                for param in stage.parameters():
                    param.requires_grad = False
        
        # Anomaly score generator
        if use_anomaly:
            self.anomaly_gen = AnomalyScoreGenerator(self.feat_dim, self.feat_dim // 4)
        
        # Scale-Adaptive Attention
        if use_saa:
            self.saa = ScaleAdaptiveAttention(self.feat_dim, num_scales=4)
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, num_classes)
        )
        
        # Auxiliary head (for anomaly features)
        if use_anomaly:
            self.aux_head = nn.Sequential(
                nn.Flatten(),
                nn.LayerNorm(self.feat_dim),
                nn.Dropout(dropout),
                nn.Linear(self.feat_dim, num_classes)
            )
        
        # Contrastive head
        if use_contrastive:
            self.contrastive_head = ContrastiveDefectHead(self.feat_dim, 128)
        
        # Initialize new modules
        self._init_new_modules()
    
    def _init_new_modules(self):
        for module in [self.head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward_backbone(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.permute(x)  # (B, H, W, C) -> (B, C, H, W)
        return x
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Extract features from pretrained backbone
        feat = self.forward_backbone(x)  # (B, C, H/32, W/32)
        
        # Generate anomaly score
        anomaly_score = None
        if self.use_anomaly:
            anomaly_score, _ = self.anomaly_gen(feat)
        
        # Apply Scale-Adaptive Attention
        scale_weights = None
        if self.use_saa:
            feat, scale_weights = self.saa(feat)
        
        # Main classification
        pooled = self.pool(feat).flatten(1)
        logits = self.head(pooled)
        
        # Auxiliary classification (from anomaly-modulated features)
        aux_logits = None
        if self.use_anomaly:
            aux_pooled = self.pool(feat * (1 + anomaly_score)).flatten(1)
            aux_logits = self.aux_head(aux_pooled)
        
        # Contrastive embeddings
        embeddings = None
        if self.use_contrastive:
            embeddings = self.contrastive_head(pooled)
        
        return {
            'logits': logits,
            'aux_logits': aux_logits,
            'embeddings': embeddings,
            'anomaly_score': anomaly_score,
            'scale_weights': scale_weights
        }
    
    def forward_simple(self, x: Tensor) -> Tensor:
        return self.forward(x)['logits']


class DefectLoCKv2Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        aux_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.contrastive_weight = contrastive_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.07)
    
    def focal_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs: Dict[str, Tensor], targets: Tensor) -> Dict[str, Tensor]:
        losses = {}
        
        losses['main'] = self.focal_loss(outputs['logits'], targets)
        
        if outputs['aux_logits'] is not None:
            losses['aux'] = F.cross_entropy(outputs['aux_logits'], targets, 
                                            label_smoothing=self.label_smoothing)
        else:
            losses['aux'] = torch.tensor(0.0, device=targets.device)
        
        if outputs['embeddings'] is not None:
            losses['contrastive'] = self.contrastive_loss(outputs['embeddings'], targets)
        else:
            losses['contrastive'] = torch.tensor(0.0, device=targets.device)
        
        losses['total'] = (
            losses['main'] + 
            self.aux_weight * losses['aux'] + 
            self.contrastive_weight * losses['contrastive']
        )
        
        return losses


def get_defect_lock_v2(
    num_classes: int = 7,
    model_size: str = 't',
    pretrained: bool = True,
    config: str = 'default',
    **kwargs
) -> DefectLoCKv2:
    configs = {
        'default': {
            'use_saa': True,
            'use_anomaly': True,
            'use_contrastive': True,
            'freeze_backbone': False,
            'dropout': 0.3
        },
        'light': {
            'use_saa': True,
            'use_anomaly': False,
            'use_contrastive': False,
            'freeze_backbone': False,
            'dropout': 0.3
        },
        'frozen': {
            'use_saa': True,
            'use_anomaly': True,
            'use_contrastive': True,
            'freeze_backbone': True,
            'dropout': 0.3
        }
    }
    
    cfg = configs.get(config, configs['default'])
    cfg.update(kwargs)
    
    return DefectLoCKv2(
        num_classes=num_classes,
        model_size=model_size,
        pretrained=pretrained,
        **cfg
    )


if __name__ == '__main__':
    print("=" * 70)
    print("DefectLoCK V2: Pretrained Backbone + Defect Innovations")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_defect_lock_v2(num_classes=7, model_size='t', config='default').to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    
    x = torch.randn(2, 3, 512, 512).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        print(f"Logits: {outputs['logits'].shape}")
        print(f"Aux logits: {outputs['aux_logits'].shape if outputs['aux_logits'] is not None else 'None'}")
        print(f"Anomaly score: {outputs['anomaly_score'].shape if outputs['anomaly_score'] is not None else 'None'}")
    
    model.train()
    outputs = model(x)
    loss_fn = DefectLoCKv2Loss(num_classes=7)
    targets = torch.randint(0, 7, (2,)).to(device)
    losses = loss_fn(outputs, targets)
    print(f"Total loss: {losses['total'].item():.4f}")
    
    print("\nTest passed!")
