#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer-based Classification Models
- Optimized for small defect detection
- Vision Transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    swin_t, swin_s, swin_b,
    Swin_T_Weights, Swin_S_Weights, Swin_B_Weights,
    swin_v2_t, swin_v2_s, swin_v2_b,
    Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights,
    maxvit_t, MaxVit_T_Weights,
    vit_b_16, vit_b_32, vit_l_16,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights
)
from typing import Optional, Tuple, List, Dict
import math


# ============================================================================
# 1. Swin Transformer V2 (Recommended for small defects)
# ============================================================================

class SwinTransformerV2Classifier(nn.Module):
    """
    Swin Transformer V2
    - Shifted Window based local attention - effective for small defect detection
    - Hierarchical feature extraction
    - Relative position encoding for various resolutions
    """
    def __init__(
        self,
        num_classes: int,
        model_size: str = 't',  # 't', 's', 'b'
        pretrained: bool = True,
        dropout: float = 0.3,
        use_multi_scale: bool = False  # Disabled for stability
    ):
        super().__init__()
        self.use_multi_scale = use_multi_scale
        
        # Backbone selection
        if model_size == 't':
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_t(weights=weights)
            feature_dim = 768
        elif model_size == 's':
            weights = Swin_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_s(weights=weights)
            feature_dim = 768
        else:  # 'b'
            weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_b(weights=weights)
            feature_dim = 1024
        
        self.feature_dim = feature_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Remove original head
        self.backbone.head = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swin V2 forward
        x = self.backbone.features(x)
        
        # x shape: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return self.classifier(x)


# ============================================================================
# 2. MaxViT (Multi-Axis Vision Transformer)
# ============================================================================

class MaxViTClassifier(nn.Module):
    """
    MaxViT - Multi-Axis Vision Transformer
    - Block attention (local) + Grid attention (global) combined
    - Captures both local features and global context
    """
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        weights = MaxVit_T_Weights.DEFAULT if pretrained else None
        self.backbone = maxvit_t(weights=weights)
        feature_dim = 512
        
        # Custom classifier
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ============================================================================
# 3. Vision Transformer (ViT) with Attention Pooling
# ============================================================================

class ViTWithAttentionPooling(nn.Module):
    """
    ViT + Learnable Attention Pooling
    - Standard ViT with attention-based pooling
    - Automatically focuses on small defect regions
    """
    def __init__(
        self,
        num_classes: int,
        model_size: str = 'b16',  # 'b16', 'b32', 'l16'
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Backbone selection
        if model_size == 'b16':
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = vit_b_16(weights=weights)
            feature_dim = 768
            num_heads = 12
        elif model_size == 'b32':
            weights = ViT_B_32_Weights.DEFAULT if pretrained else None
            self.backbone = vit_b_32(weights=weights)
            feature_dim = 768
            num_heads = 12
        else:  # 'l16'
            weights = ViT_L_16_Weights.DEFAULT if pretrained else None
            self.backbone = vit_l_16(weights=weights)
            feature_dim = 1024
            num_heads = 16
        
        self.feature_dim = feature_dim
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(feature_dim, num_heads)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Remove original head
        self.backbone.heads = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get patch embeddings (excluding CLS token)
        x = self.backbone._process_input(x)
        n = x.shape[0]
        
        # Add class token
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Transformer encoder
        x = self.backbone.encoder(x)
        
        # Attention pooling (use all tokens including CLS)
        x = self.attention_pool(x)
        
        return self.classifier(x)


class AttentionPooling(nn.Module):
    """Learnable Attention Pooling"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Query from learnable parameter
        q = self.query.expand(B, -1, -1)
        q = q.reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Key, Value from input
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        
        return x.squeeze(1)


# ============================================================================
# 4. Transformer Block (for custom models)
# ============================================================================

class TransformerBlock(nn.Module):
    """Standard Transformer Block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 5. PVTv2 - Pyramid Vision Transformer V2
# ============================================================================

class PVTv2Classifier(nn.Module):
    """
    PVTv2 - Pyramid Vision Transformer V2
    - Multi-scale feature extraction (like CNN)
    - Spatial Reduction Attention for efficiency
    - Very effective for small defect detection
    """
    def __init__(
        self,
        num_classes: int,
        img_size: int = 384,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [3, 4, 6, 3],
        sr_ratios: List[int] = [8, 4, 2, 1],
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_stages = 4
        
        # Patch embeddings for each stage
        self.patch_embeds = nn.ModuleList()
        self.patch_embeds.append(
            PatchEmbed(3, embed_dims[0], patch_size=4, stride=4)
        )
        for i in range(1, self.num_stages):
            self.patch_embeds.append(
                PatchEmbed(embed_dims[i-1], embed_dims[i], patch_size=2, stride=2)
            )
        
        # Transformer stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.ModuleList([
                SRAttentionBlock(
                    embed_dims[i], num_heads[i], mlp_ratio, 
                    sr_ratio=sr_ratios[i], dropout=dropout
                )
                for _ in range(depths[i])
            ])
            self.stages.append(stage)
        
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in embed_dims])
        
        # Classifier (use final stage only for simplicity)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dims[-1]),
            nn.Dropout(dropout),
            nn.Linear(embed_dims[-1], 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        for i in range(self.num_stages):
            # Patch embedding
            x, H, W = self.patch_embeds[i](x)
            
            # Transformer blocks
            for block in self.stages[i]:
                x = block(x, H, W)
            
            x = self.norms[i](x)
            
            # Reshape for next stage (except last)
            if i < self.num_stages - 1:
                x = x.transpose(1, 2).view(B, -1, H, W)
        
        # Classify using final features
        x = x.transpose(1, 2)  # [B, C, N]
        return self.classifier(x)


class PatchEmbed(nn.Module):
    """Patch Embedding with overlapping"""
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=stride, 
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SRAttentionBlock(nn.Module):
    """Spatial Reduction Attention Block"""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class SRAttention(nn.Module):
    """Spatial Reduction Attention"""
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).view(B, C, H, W)
            x_ = self.sr(x_).flatten(2).transpose(1, 2)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


# ============================================================================
# 6. EfficientViT (Lightweight Vision Transformer)
# ============================================================================

class EfficientViTClassifier(nn.Module):
    """
    EfficientViT - Lightweight Vision Transformer
    - Sandwich Layout (Conv-Transformer-Conv)
    - Fast inference speed
    - Suitable for real-time inspection
    """
    def __init__(
        self,
        num_classes: int,
        img_size: int = 384,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Stem (Conv layers)
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        self.num_patches = (img_size // 4) ** 2
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Efficient Transformer blocks
        self.blocks = nn.ModuleList([
            EfficientTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        
        # Flatten and add position
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Handle variable input sizes
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=x.size(1),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
        
        x = x + pos_embed
        
        # Transformer
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Head
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.head(x)
        
        return x


class EfficientTransformerBlock(nn.Module):
    """Efficient Transformer Block with Local-Global attention"""
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        # Local (depthwise conv)
        self.local = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        
        # Global (lightweight attention)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local
        local_out = self.local(x.transpose(1, 2)).transpose(1, 2)
        x = x + local_out
        
        # Global attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


# ============================================================================
# 7. Hybrid CNN-Transformer
# ============================================================================

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN-Transformer
    - CNN for low-level features (edges, textures)
    - Transformer for high-level relationships
    - Captures both local and global context
    """
    def __init__(
        self,
        num_classes: int,
        cnn_backbone: str = 'resnet50',
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 6,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        
        # CNN backbone (feature extractor)
        from torchvision.models import resnet50, ResNet50_Weights
        cnn = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Use layers up to layer3 (1/16 resolution)
        self.cnn_features = nn.Sequential(
            cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool,
            cnn.layer1, cnn.layer2, cnn.layer3
        )
        cnn_out_dim = 1024  # ResNet layer3 output
        
        # Project to transformer dimension
        self.proj = nn.Conv2d(cnn_out_dim, embed_dim, 1)
        
        # Learnable position embedding (will be interpolated for different sizes)
        self.pos_embed = nn.Parameter(torch.zeros(1, 576, embed_dim))  # 24x24 for 384 input
        
        # Transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features
        x = self.cnn_features(x)
        x = self.proj(x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Interpolate position embedding if needed
        N = x.size(1)
        if N != self.pos_embed.size(1):
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
        
        x = x + pos_embed
        
        # Transformer
        for block in self.transformer:
            x = block(x)
        
        x = self.norm(x)
        
        # Classify
        x = x.transpose(1, 2)
        return self.classifier(x)


# ============================================================================
# 8. Factory Function
# ============================================================================

def get_transformer_classifier(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Model Selection Guide:
    
    1. 'swin_v2' (Highly Recommended)
       - Most effective for small defect detection
       - Local attention for fine-grained features
       - Computationally efficient
    
    2. 'maxvit'
       - Local + Global attention combined
       - Balanced performance
       - Good for medium-sized defects too
    
    3. 'vit_attention_pool'
       - Strong global context learning
       - Attention pooling for important regions
       - Finding small defects in large images
    
    4. 'pvtv2'
       - Multi-scale feature extraction
       - Detects defects of various sizes
       - Dense prediction capable
    
    5. 'efficient_vit'
       - Fast inference speed
       - Suitable for real-time inspection
       - Edge device deployment
    
    6. 'hybrid_cnn_transformer'
       - CNN + Transformer advantages combined
       - Low-level features + high-level relationships
    """
    
    models_dict = {
        'swin_v2': SwinTransformerV2Classifier,
        'maxvit': MaxViTClassifier,
        'vit_attention_pool': ViTWithAttentionPooling,
        'pvtv2': PVTv2Classifier,
        'efficient_vit': EfficientViTClassifier,
        'hybrid_cnn_transformer': HybridCNNTransformer,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models_dict.keys())}")
    
    return models_dict[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)


# ============================================================================
# 9. Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss - effective for class imbalance"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing - improves generalization"""
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.full_like(pred, smooth_value)
        one_hot.scatter_(1, target.unsqueeze(1), confidence)
        
        log_prob = F.log_softmax(pred, dim=1)
        return (-one_hot * log_prob).sum(dim=1).mean()

# ============================================================================
# SAM3 Vision Encoder for Classification
# ============================================================================

class SAM3Classifier(nn.Module):
    """
    SAM3 Vision Encoder (Hiera backbone) for Classification
    - SAM3's Perception Encoder as feature extractor
    - Requires: pip install transformers
    - Requires: Hugging Face access token for SAM3 checkpoint
    
    Note: SAM3 checkpoint requires access request at:
    https://huggingface.co/facebook/sam3
    """
    def __init__(
        self,
        num_classes: int,
        model_name: str = "facebook/sam3-hiera-large",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        try:
            from transformers import Sam3Model, Sam3Config
            
            if pretrained:
                self.sam3 = Sam3Model.from_pretrained(model_name)
            else:
                config = Sam3Config()
                self.sam3 = Sam3Model(config)
            
            # Get vision encoder output dimension
            # SAM3 Hiera large: 1024, tiny: 768
            if "large" in model_name:
                feature_dim = 1024
            elif "base" in model_name:
                feature_dim = 768
            else:  # tiny
                feature_dim = 768
            
            self.feature_dim = feature_dim
            
            # Freeze encoder if specified
            if freeze_encoder:
                for param in self.sam3.vision_encoder.parameters():
                    param.requires_grad = False
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 512),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(512, num_classes)
            )
            
            self.use_transformers = True
            
        except ImportError:
            print("transformers library not found. Using fallback Hiera implementation.")
            self.use_transformers = False
            self._build_hiera_fallback(num_classes, dropout)
    
    def _build_hiera_fallback(self, num_classes: int, dropout: float):
        """Fallback: Build Hiera-like architecture manually"""
        # Simplified Hiera-like architecture
        self.patch_embed = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        
        # Hiera stages
        self.stages = nn.ModuleList([
            HieraStage(96, 96, depth=1, num_heads=1, window_size=8),
            HieraStage(96, 192, depth=2, num_heads=2, window_size=8, downsample=True),
            HieraStage(192, 384, depth=11, num_heads=4, window_size=8, downsample=True),
            HieraStage(384, 768, depth=2, num_heads=8, window_size=8, downsample=True),
        ])
        
        self.norm = nn.LayerNorm(768)
        self.feature_dim = 768
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_transformers:
            # Use SAM3 vision encoder
            outputs = self.sam3.vision_encoder(x)
            # Get the last hidden state
            features = outputs.last_hidden_state  # [B, H*W, C]
            
            # Reshape to spatial format
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).view(B, C, H, W)
            
            return self.classifier(features)
        else:
            # Fallback path
            x = self.patch_embed(x)
            
            for stage in self.stages:
                x = stage(x)
            
            # x: [B, C, H, W]
            return self.classifier(x)


class HieraStage(nn.Module):
    """Simplified Hiera Stage"""
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        depth: int, 
        num_heads: int,
        window_size: int = 8,
        downsample: bool = False
    ):
        super().__init__()
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.LayerNorm([out_dim, 1, 1])  # Will be applied after reshape
            )
            in_dim = out_dim
        
        self.blocks = nn.ModuleList([
            HieraBlock(in_dim if i == 0 and not downsample else out_dim, 
                      out_dim, num_heads, window_size)
            for i in range(depth)
        ])
        
        self.out_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            x = self.downsample[0](x)  # Conv
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            x = F.layer_norm(x, [C])
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        for block in self.blocks:
            x = block(x)
        
        return x


class HieraBlock(nn.Module):
    """Simplified Hiera Block with Window Attention"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, window_size: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(out_dim)
        self.attn = WindowAttention(out_dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim)
        )
        
        self.proj = nn.Identity()
        if in_dim != out_dim:
            self.proj = nn.Conv2d(in_dim, out_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        
        # Reshape for attention
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.view(B, H * W, C)
        
        # Attention
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention"""
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        
        # Simple global attention for now (can be optimized with window partitioning)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


# ============================================================================
# 10. Test
# ============================================================================

if __name__ == "__main__":
    # Settings
    num_classes = 4
    batch_size = 2
    img_size = 384
    
    # Dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print("=" * 70)
    print("Transformer Classification Models Test")
    print("=" * 70)
    
    # Test models
    model_configs = [
        ('swin_v2', {'model_size': 't'}),
        ('maxvit', {}),
        ('efficient_vit', {'img_size': img_size}),
    ]
    
    for model_name, kwargs in model_configs:
        print(f"\n{model_name}:")
        try:
            model = get_transformer_classifier(
                model_name, num_classes, pretrained=False, **kwargs
            )
            
            # Parameter count
            params = sum(p.numel() for p in model.parameters()) / 1e6
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  Parameters: {params:.2f}M")
            print(f"  Input: {dummy_input.shape}")
            print(f"  Output: {output.shape}")
            print(f"  OK")
            
        except Exception as e:
            print(f"  Error: {e}")