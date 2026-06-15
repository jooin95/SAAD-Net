#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    resnet50, ResNet50_Weights
)
from typing import Optional, Tuple, List, Dict, Union
import numpy as np


# ============================================================================
# 1. Feature Extractor
# ============================================================================

class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor for high-resolution images"""
    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: List[str] = ['layer1', 'layer2', 'layer3'],
        pretrained: bool = True
    ):
        super().__init__()
        self.layers = layers
        
        if backbone == 'wide_resnet50_2':
            weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            self.backbone = wide_resnet50_2(weights=weights)
            self.feature_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        elif backbone == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = resnet50(weights=weights)
            self.feature_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if 'layer1' in self.layers:
            features['layer1'] = x
        
        x = self.backbone.layer2(x)
        if 'layer2' in self.layers:
            features['layer2'] = x
        
        x = self.backbone.layer3(x)
        if 'layer3' in self.layers:
            features['layer3'] = x
        
        x = self.backbone.layer4(x)
        if 'layer4' in self.layers:
            features['layer4'] = x
        
        return features


# ============================================================================
# 2. Reverse Distillation (RD4AD)
# ============================================================================

class OCBEDecoder(nn.Module):
    """One-Class Bottleneck Embedding Decoder"""
    def __init__(self, feature_dims: Dict[str, int], layers: List[str], bottleneck_dim: int = 256):
        super().__init__()
        self.layers = layers
        
        self.decoders = nn.ModuleDict()
        for layer in layers:
            dim = feature_dims[layer]
            self.decoders[layer] = nn.Sequential(
                nn.Conv2d(bottleneck_dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 1)
            )
    
    def forward(self, bottleneck: torch.Tensor, target_sizes: Dict[str, Tuple[int, int]]) -> Dict[str, torch.Tensor]:
        outputs = {}
        for layer in self.layers:
            x = F.interpolate(bottleneck, size=target_sizes[layer], mode='bilinear', align_corners=False)
            outputs[layer] = self.decoders[layer](x)
        return outputs


class ReverseDistillation(nn.Module):
    """
    Reverse Distillation for Anomaly Detection
    """
    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: List[str] = ['layer1', 'layer2', 'layer3'],
        bottleneck_dim: int = 256,
        pretrained: bool = True
    ):
        super().__init__()
        self.layers = layers
        
        # Teacher (frozen)
        self.teacher = MultiScaleFeatureExtractor(backbone, layers, pretrained)
        
        # Student encoder (trainable)
        self.student = MultiScaleFeatureExtractor(backbone, layers, pretrained=False)
        for param in self.student.parameters():
            param.requires_grad = True
        
        # Bottleneck
        total_dim = sum(self.teacher.feature_dims[l] for l in layers)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_dim, bottleneck_dim, 1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = OCBEDecoder(self.teacher.feature_dims, layers, bottleneck_dim)
        self.feature_dims = self.teacher.feature_dims
    
    def forward(self, x: torch.Tensor, return_map: bool = False):
        with torch.no_grad():
            teacher_features = self.teacher(x)
        
        student_features = self.student(x)
        target_sizes = {l: teacher_features[l].shape[-2:] for l in self.layers}
        
        # Bottleneck
        concat_features = []
        min_size = min(target_sizes[l][0] for l in self.layers)
        for layer in self.layers:
            feat = student_features[layer]
            if feat.shape[-1] != min_size:
                feat = F.interpolate(feat, size=(min_size, min_size), mode='bilinear', align_corners=False)
            concat_features.append(feat)
        
        concat = torch.cat(concat_features, dim=1)
        bottleneck = self.bottleneck(concat)
        decoded_features = self.decoder(bottleneck, target_sizes)
        
        # Anomaly maps
        anomaly_maps = []
        for layer in self.layers:
            t_feat = F.normalize(teacher_features[layer], p=2, dim=1)
            d_feat = F.normalize(decoded_features[layer], p=2, dim=1)
            anomaly = 1 - (t_feat * d_feat).sum(dim=1, keepdim=True)
            anomaly_maps.append(anomaly)
        
        # Combine
        max_size = max(m.shape[-1] for m in anomaly_maps)
        resized = [F.interpolate(m, size=(max_size, max_size), mode='bilinear', align_corners=False) if m.shape[-1] != max_size else m for m in anomaly_maps]
        combined_map = torch.stack(resized, dim=0).mean(dim=0)
        
        anomaly_score = torch.amax(combined_map, dim=(1, 2, 3))
        
        if return_map:
            return anomaly_score, combined_map
        return anomaly_score
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_features = self.teacher(x)
        
        student_features = self.student(x)
        target_sizes = {l: teacher_features[l].shape[-2:] for l in self.layers}
        
        concat_features = []
        min_size = min(target_sizes[l][0] for l in self.layers)
        for layer in self.layers:
            feat = student_features[layer]
            if feat.shape[-1] != min_size:
                feat = F.interpolate(feat, size=(min_size, min_size), mode='bilinear', align_corners=False)
            concat_features.append(feat)
        
        concat = torch.cat(concat_features, dim=1)
        bottleneck = self.bottleneck(concat)
        decoded_features = self.decoder(bottleneck, target_sizes)
        
        total_loss = 0
        for layer in self.layers:
            t_feat = F.normalize(teacher_features[layer], p=2, dim=1)
            d_feat = F.normalize(decoded_features[layer], p=2, dim=1)
            loss = 1 - (t_feat * d_feat).sum(dim=1).mean()
            total_loss = total_loss + loss
        
        return total_loss / len(self.layers)


# ============================================================================
# 3. FastFlow
# ============================================================================

class CouplingBlock(nn.Module):
    """Coupling block for normalizing flow"""
    def __init__(self, channels: int, subnet_channels: int = 256):
        super().__init__()
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        
        self.subnet1 = nn.Sequential(
            nn.Conv2d(self.split_len1, subnet_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(subnet_channels, self.split_len2 * 2, 3, padding=1)
        )
        self.subnet2 = nn.Sequential(
            nn.Conv2d(self.split_len2, subnet_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(subnet_channels, self.split_len1 * 2, 3, padding=1)
        )
        
        nn.init.zeros_(self.subnet1[-1].weight)
        nn.init.zeros_(self.subnet1[-1].bias)
        nn.init.zeros_(self.subnet2[-1].weight)
        nn.init.zeros_(self.subnet2[-1].bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x[:, :self.split_len1], x[:, self.split_len1:]
        
        s2, t2 = self.subnet1(x1).chunk(2, dim=1)
        s2 = torch.sigmoid(s2 + 2) * 2
        y2 = (x2 + t2) * s2
        
        s1, t1 = self.subnet2(y2).chunk(2, dim=1)
        s1 = torch.sigmoid(s1 + 2) * 2
        y1 = (x1 + t1) * s1
        
        log_det = torch.sum(torch.log(s1), dim=(1, 2, 3)) + torch.sum(torch.log(s2), dim=(1, 2, 3))
        return torch.cat([y1, y2], dim=1), log_det


class FastFlow(nn.Module):
    """FastFlow: Normalizing Flow for Anomaly Detection"""
    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: List[str] = ['layer2', 'layer3'],
        flow_steps: int = 8,
        pretrained: bool = True
    ):
        super().__init__()
        self.layers = layers
        
        self.feature_extractor = MultiScaleFeatureExtractor(backbone, layers, pretrained)
        
        self.flows = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        for layer in layers:
            dim = self.feature_extractor.feature_dims[layer]
            self.norms[layer] = nn.BatchNorm2d(dim, affine=False)
            self.flows[layer] = nn.ModuleList([CouplingBlock(dim) for _ in range(flow_steps)])
    
    def forward(self, x: torch.Tensor, return_map: bool = False):
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        anomaly_maps = []
        for layer in self.layers:
            feat = self.norms[layer](features[layer])
            
            z, log_det_sum = feat, 0
            for block in self.flows[layer]:
                z, log_det = block(z)
                log_det_sum = log_det_sum + log_det
            
            log_prob = -0.5 * torch.sum(z ** 2, dim=1, keepdim=True) + log_det_sum.view(-1, 1, 1, 1)
            anomaly_maps.append(-log_prob)
        
        max_size = max(m.shape[-1] for m in anomaly_maps)
        resized = [F.interpolate(m, size=(max_size, max_size), mode='bilinear', align_corners=False) if m.shape[-1] != max_size else m for m in anomaly_maps]
        combined_map = torch.stack(resized, dim=0).mean(dim=0)
        
        B = combined_map.shape[0]
        combined_map = combined_map.view(B, -1)
        combined_map = (combined_map - combined_map.mean(dim=1, keepdim=True)) / (combined_map.std(dim=1, keepdim=True) + 1e-6)
        combined_map = combined_map.view(B, 1, max_size, max_size)
        
        anomaly_score = torch.amax(combined_map, dim=(1, 2, 3))
        
        if return_map:
            return anomaly_score, combined_map
        return anomaly_score
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        total_loss = 0
        for layer in self.layers:
            feat = self.norms[layer](features[layer])
            
            z, log_det_sum = feat, 0
            for block in self.flows[layer]:
                z, log_det = block(z)
                log_det_sum = log_det_sum + log_det
            
            nll = 0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) - log_det_sum
            total_loss = total_loss + nll.mean()
        
        return total_loss / len(self.layers)


# ============================================================================
# 4. DRAEM
# ============================================================================

class UNetReconstructor(nn.Module):
    """U-Net for reconstruction"""
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        
        self.enc1 = self._block(in_ch, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.enc4 = self._block(base_ch * 4, base_ch * 8)
        self.bottleneck = self._block(base_ch * 8, base_ch * 16)
        
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = self._block(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = self._block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = self._block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = self._block(base_ch * 2, base_ch)
        self.final = nn.Conv2d(base_ch, in_ch, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)


class DRAEM(nn.Module):
    """DRAEM: Reconstruction with Synthetic Anomalies"""
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.reconstructor = UNetReconstructor(3, base_channels)
        self.segmentor = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, return_map: bool = False):
        recon = self.reconstructor(x)
        mask = self.segmentor(torch.cat([x, recon], dim=1))
        anomaly_score = torch.amax(mask, dim=(1, 2, 3))
        
        if return_map:
            return anomaly_score, mask
        return anomaly_score
    
    def compute_loss(self, x_normal: torch.Tensor, x_anomaly: torch.Tensor, mask_gt: torch.Tensor):
        recon = self.reconstructor(x_anomaly)
        recon_loss = F.mse_loss(recon, x_normal)
        
        pred_mask = self.segmentor(torch.cat([x_anomaly, recon], dim=1))
        seg_loss = F.binary_cross_entropy(pred_mask, mask_gt)
        
        return recon_loss + seg_loss


# ============================================================================
# 5. Factory Function
# ============================================================================

def get_highres_anomaly_detector(model_name: str, **kwargs) -> nn.Module:
    """
    Available models:
    """
    models = {
        'reverse_distillation': ReverseDistillation,
        'rd4ad': ReverseDistillation,
        'fastflow': FastFlow,
        'draem': DRAEM,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == "__main__":
    print("=" * 70)
    print("High-Resolution Anomaly Detection Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1024x1024
    print("\nTesting with 1024x1024 input:")
    dummy = torch.randn(1, 3, 1024, 1024).to(device)
    
    model = ReverseDistillation().to(device)
    model.eval()
    with torch.no_grad():
        score, amap = model(dummy, return_map=True)
    print(f"ReverseDistillation - Score: {score.shape}, Map: {amap.shape}")
    
    print("\nAll tests passed!")
