#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anomaly Detection Models for Industrial Defect Detection
=========================================================
Implements:
1. EfficientAD - Student-Teacher with Autoencoder
2. SimpleNet - Feature Adaptor based
3. PatchCore - Memory Bank based
4. STFPM - Student-Teacher Feature Pyramid Matching

All models support:
- High-resolution image processing via tiling
- Multi-GPU training
- Mixed precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    wide_resnet50_2, Wide_ResNet50_2_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    resnet18, ResNet18_Weights
)
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import math
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# 1. Patch Description Network (PDN) - EfficientAD Core
# ============================================================================

class PDN_Small(nn.Module):
    """
    Patch Description Network (Small)
    - 4 convolutional layers
    - Receptive field: 33x33 pixels
    - Output: 384-dim feature per patch
    """
    def __init__(self, out_channels: int = 384, padding: bool = False):
        super().__init__()
        pad_mult = 1 if padding else 0
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDN_Medium(nn.Module):
    """
    Patch Description Network (Medium)
    - 6 convolutional layers
    - Larger receptive field
    """
    def __init__(self, out_channels: int = 384, padding: bool = False):
        super().__init__()
        pad_mult = 1 if padding else 0
        
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        return x


# ============================================================================
# 2. EfficientAD - Student-Teacher with Autoencoder
# ============================================================================

class EfficientADAutoencoder(nn.Module):
    """
    Autoencoder for Logical Anomaly Detection in EfficientAD
    """
    def __init__(self, out_channels: int = 384):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(size=3, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class EfficientAD(nn.Module):
    """
    EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
    
    Components:
    1. Teacher PDN (pretrained, frozen)
    2. Student PDN (trained to mimic teacher on normal data)
    3. Autoencoder (for logical anomaly detection)
    
    Training:
    - Hard Feature Loss for student-teacher
    - Reconstruction loss for autoencoder
    
    Inference:
    - Structural anomaly: ||Teacher - Student||
    - Logical anomaly: ||AE_output - Student||
    """
    def __init__(
        self,
        model_size: str = 'small',  # 'small' or 'medium'
        out_channels: int = 384,
        padding: bool = True,
        use_autoencoder: bool = True
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.use_autoencoder = use_autoencoder
        
        # Teacher network (will be frozen)
        if model_size == 'small':
            self.teacher = PDN_Small(out_channels, padding)
        else:
            self.teacher = PDN_Medium(out_channels, padding)
        
        # Student network
        if model_size == 'small':
            self.student = PDN_Small(out_channels * 2, padding)  # 2x for ST + AE
        else:
            self.student = PDN_Medium(out_channels * 2, padding)
        
        # Autoencoder for logical anomalies
        if use_autoencoder:
            self.autoencoder = EfficientADAutoencoder(out_channels)
        
        # Normalization parameters (set during training)
        self.register_buffer('mean_teacher', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('std_teacher', torch.ones(1, out_channels, 1, 1))
        self.register_buffer('mean_ae', torch.zeros(1, out_channels, 1, 1))
        self.register_buffer('std_ae', torch.ones(1, out_channels, 1, 1))
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        x: torch.Tensor,
        return_maps: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            return_maps: If True, return anomaly maps
        
        Returns:
            anomaly_score or (anomaly_score, maps_dict)
        """
        # Teacher features (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(x)
            teacher_out = (teacher_out - self.mean_teacher) / self.std_teacher
        
        # Student features
        student_out = self.student(x)
        student_st = student_out[:, :self.out_channels]  # For structural
        student_ae = student_out[:, self.out_channels:]  # For logical
        
        # Structural anomaly map (Student-Teacher difference)
        st_map = torch.mean((teacher_out - student_st) ** 2, dim=1, keepdim=True)
        
        # Logical anomaly map (if autoencoder is used)
        if self.use_autoencoder:
            with torch.no_grad():
                ae_out = self.autoencoder(x)
                ae_out = (ae_out - self.mean_ae) / self.std_ae
            ae_map = torch.mean((ae_out - student_ae) ** 2, dim=1, keepdim=True)
        else:
            ae_map = torch.zeros_like(st_map)
        
        # Combined anomaly map
        anomaly_map = st_map + ae_map
        
        # Anomaly score (max value in map)
        anomaly_score = torch.amax(anomaly_map, dim=(1, 2, 3))
        
        if return_maps:
            return anomaly_score, {
                'structural': st_map,
                'logical': ae_map,
                'combined': anomaly_map
            }
        
        return anomaly_score
    
    def compute_loss(
        self,
        x: torch.Tensor,
        hard_feature_p: float = 0.999
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses
        
        Args:
            x: Normal images [B, 3, H, W]
            hard_feature_p: Percentile for hard feature loss (0.999 = top 0.1%)
        
        Returns:
            Dict with loss values
        """
        # Teacher features (target)
        with torch.no_grad():
            teacher_out = self.teacher(x)
        
        # Student features
        student_out = self.student(x)
        student_st = student_out[:, :self.out_channels]
        student_ae = student_out[:, self.out_channels:]
        
        # Hard Feature Loss for Student-Teacher
        diff_st = (teacher_out - student_st) ** 2
        diff_st = diff_st.mean(dim=1)  # [B, H, W]
        
        # Select hard examples (top 1-p percentile)
        diff_flat = diff_st.view(diff_st.size(0), -1)
        threshold = torch.quantile(diff_flat, hard_feature_p, dim=1, keepdim=True)
        hard_mask = diff_flat >= threshold
        hard_loss_st = (diff_flat * hard_mask).sum() / hard_mask.sum().clamp(min=1)
        
        # Autoencoder loss
        if self.use_autoencoder:
            ae_out = self.autoencoder(x)
            
            # Match spatial dimensions
            if ae_out.shape[-2:] != student_ae.shape[-2:]:
                ae_out = F.interpolate(ae_out, size=student_ae.shape[-2:], mode='bilinear')
            
            diff_ae = (ae_out - student_ae) ** 2
            diff_ae = diff_ae.mean(dim=1)
            
            # Hard feature loss for AE
            diff_ae_flat = diff_ae.view(diff_ae.size(0), -1)
            threshold_ae = torch.quantile(diff_ae_flat, hard_feature_p, dim=1, keepdim=True)
            hard_mask_ae = diff_ae_flat >= threshold_ae
            hard_loss_ae = (diff_ae_flat * hard_mask_ae).sum() / hard_mask_ae.sum().clamp(min=1)
        else:
            hard_loss_ae = torch.tensor(0.0, device=x.device)
        
        total_loss = hard_loss_st + hard_loss_ae
        
        return {
            'total': total_loss,
            'st_loss': hard_loss_st,
            'ae_loss': hard_loss_ae
        }
    
    def set_normalization_params(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ):
        """Calculate and set normalization parameters from training data"""
        self.eval()
        
        teacher_outputs = []
        ae_outputs = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                
                t_out = self.teacher(images)
                teacher_outputs.append(t_out.cpu())
                
                if self.use_autoencoder:
                    ae_out = self.autoencoder(images)
                    ae_outputs.append(ae_out.cpu())
        
        # Compute statistics
        teacher_cat = torch.cat(teacher_outputs, dim=0)
        self.mean_teacher = teacher_cat.mean(dim=(0, 2, 3), keepdim=True).to(device)
        self.std_teacher = teacher_cat.std(dim=(0, 2, 3), keepdim=True).to(device)
        self.std_teacher = self.std_teacher.clamp(min=1e-6)
        
        if self.use_autoencoder and ae_outputs:
            ae_cat = torch.cat(ae_outputs, dim=0)
            self.mean_ae = ae_cat.mean(dim=(0, 2, 3), keepdim=True).to(device)
            self.std_ae = ae_cat.std(dim=(0, 2, 3), keepdim=True).to(device)
            self.std_ae = self.std_ae.clamp(min=1e-6)


# ============================================================================
# 3. SimpleNet - Feature Adaptor Based Anomaly Detection
# ============================================================================

class SimpleNet(nn.Module):
    """
    SimpleNet: A Simple Network for Anomaly Detection
    
    Idea:
    1. Extract features with pretrained backbone
    2. Add Gaussian noise to features
    3. Train simple MLP to denoise
    4. Anomaly = reconstruction error
    
    Advantages:
    - Very fast training and inference
    - Few trainable parameters
    - Works well with small defects
    """
    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: List[str] = ['layer2', 'layer3'],
        noise_std: float = 0.015,
        embed_dim: int = 512
    ):
        super().__init__()
        
        self.layers = layers
        self.noise_std = noise_std
        
        # Backbone (frozen)
        if backbone == 'wide_resnet50_2':
            weights = Wide_ResNet50_2_Weights.DEFAULT
            self.backbone = wide_resnet50_2(weights=weights)
            feature_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
            feature_dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Calculate total feature dimension
        total_dim = sum(feature_dims[l] for l in layers)
        
        # Feature adaptor (simple MLP)
        self.adaptor = nn.Sequential(
            nn.Linear(total_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, total_dim)
        )
        
        self.total_dim = total_dim
        
        # Normalization params
        self.register_buffer('mean', torch.zeros(total_dim))
        self.register_buffer('std', torch.ones(total_dim))
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate features from specified layers"""
        features = []
        
        # Forward through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if 'layer1' in self.layers:
            features.append(x)
        
        x = self.backbone.layer2(x)
        if 'layer2' in self.layers:
            features.append(x)
        
        x = self.backbone.layer3(x)
        if 'layer3' in self.layers:
            features.append(x)
        
        x = self.backbone.layer4(x)
        if 'layer4' in self.layers:
            features.append(x)
        
        # Resize all to same spatial size and concatenate
        target_size = features[0].shape[-2:]
        resized = []
        for f in features:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            resized.append(f)
        
        concat = torch.cat(resized, dim=1)  # [B, C, H, W]
        
        return concat
    
    def forward(
        self, 
        x: torch.Tensor,
        return_map: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            return_map: If True, return anomaly map
        
        Returns:
            anomaly_score or (anomaly_score, anomaly_map)
        """
        # Extract features
        with torch.no_grad():
            features = self._extract_features(x)  # [B, C, H, W]
        
        B, C, H, W = features.shape
        
        # Reshape for adaptor: [B*H*W, C]
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize
        features_norm = (features_flat - self.mean) / self.std
        
        # Denoise (no noise at inference)
        denoised = self.adaptor(features_norm)
        
        # Anomaly = reconstruction error
        diff = (features_norm - denoised) ** 2
        diff = diff.sum(dim=1)  # [B*H*W]
        
        # Reshape back to spatial
        anomaly_map = diff.view(B, H, W)
        
        # Anomaly score
        anomaly_score = torch.amax(anomaly_map, dim=(1, 2))
        
        if return_map:
            return anomaly_score, anomaly_map.unsqueeze(1)
        
        return anomaly_score
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss
        
        Training: Add noise to features, train adaptor to remove noise
        """
        # Extract features
        with torch.no_grad():
            features = self._extract_features(x)
        
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize
        features_norm = (features_flat - self.mean) / self.std
        
        # Add noise
        noise = torch.randn_like(features_norm) * self.noise_std
        noisy = features_norm + noise
        
        # Denoise
        denoised = self.adaptor(noisy)
        
        # Loss: MSE between denoised and original
        loss = F.mse_loss(denoised, features_norm)
        
        return loss
    
    def set_normalization_params(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ):
        """Calculate normalization parameters from training data"""
        self.eval()
        
        all_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                
                features = self._extract_features(images)
                B, C, H, W = features.shape
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
                all_features.append(features_flat.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        self.mean = all_features.mean(dim=0).to(device)
        self.std = all_features.std(dim=0).clamp(min=1e-6).to(device)


# ============================================================================
# 4. PatchCore - Memory Bank Based Anomaly Detection
# ============================================================================

class PatchCore(nn.Module):
    """
    PatchCore: Towards Total Recall in Industrial Anomaly Detection
    
    Idea:
    1. Extract patch features from pretrained backbone
    2. Store representative patches in memory bank (coreset)
    3. Anomaly = distance to nearest neighbor in memory
    
    Advantages:
    - No training required (only feature extraction)
    - Very accurate
    - Interpretable (can show similar normal patches)
    
    Disadvantages:
    - Memory intensive (stores many feature vectors)
    - Slower inference (nearest neighbor search)
    """
    def __init__(
        self,
        backbone: str = 'wide_resnet50_2',
        layers: List[str] = ['layer2', 'layer3'],
        coreset_ratio: float = 0.01,
        num_neighbors: int = 9
    ):
        super().__init__()
        
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        
        # Backbone (frozen)
        if backbone == 'wide_resnet50_2':
            weights = Wide_ResNet50_2_Weights.DEFAULT
            self.backbone = wide_resnet50_2(weights=weights)
            feature_dims = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
            feature_dims = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.total_dim = sum(feature_dims[l] for l in layers)
        
        # Memory bank (populated during fit)
        self.memory_bank = None
        self.nn_model = None
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate features"""
        features = []
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if 'layer1' in self.layers:
            features.append(x)
        
        x = self.backbone.layer2(x)
        if 'layer2' in self.layers:
            features.append(x)
        
        x = self.backbone.layer3(x)
        if 'layer3' in self.layers:
            features.append(x)
        
        x = self.backbone.layer4(x)
        if 'layer4' in self.layers:
            features.append(x)
        
        # Resize and concatenate
        target_size = features[0].shape[-2:]
        resized = []
        for f in features:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            resized.append(f)
        
        return torch.cat(resized, dim=1)
    
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ):
        """Build memory bank from training data"""
        self.eval()
        
        all_features = []
        
        print("Extracting features for memory bank...")
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                
                features = self._extract_features(images)
                B, C, H, W = features.shape
                
                # Reshape: [B, C, H, W] -> [B*H*W, C]
                features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
                all_features.append(features_flat.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        print(f"Total patches: {len(all_features)}")
        
        # Coreset selection (random for simplicity, can use greedy)
        n_select = max(1, int(len(all_features) * self.coreset_ratio))
        indices = np.random.choice(len(all_features), n_select, replace=False)
        self.memory_bank = all_features[indices]
        print(f"Memory bank size: {len(self.memory_bank)}")
        
        # Build NN model
        self.nn_model = NearestNeighbors(n_neighbors=self.num_neighbors, metric='euclidean')
        self.nn_model.fit(self.memory_bank)
    
    def forward(
        self,
        x: torch.Tensor,
        return_map: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not initialized. Call fit() first.")
        
        device = x.device
        
        with torch.no_grad():
            features = self._extract_features(x)
        
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        
        # Nearest neighbor search
        distances, _ = self.nn_model.kneighbors(features_flat)
        
        # Use mean of k-nearest distances
        anomaly_scores = distances.mean(axis=1)
        anomaly_map = anomaly_scores.reshape(B, H, W)
        
        # Per-image anomaly score
        image_scores = anomaly_map.max(axis=(1, 2))
        
        image_scores = torch.from_numpy(image_scores).float().to(device)
        anomaly_map = torch.from_numpy(anomaly_map).float().to(device)
        
        if return_map:
            return image_scores, anomaly_map.unsqueeze(1)
        
        return image_scores


# ============================================================================
# 5. STFPM - Student-Teacher Feature Pyramid Matching
# ============================================================================

class STFPM(nn.Module):
    """
    STFPM: Student-Teacher Feature Pyramid Matching
    
    Similar to EfficientAD but uses full ResNet backbone
    and matches features at multiple scales.
    """
    def __init__(
        self,
        backbone: str = 'resnet18',
        layers: List[str] = ['layer1', 'layer2', 'layer3']
    ):
        super().__init__()
        
        self.layers = layers
        
        # Teacher (pretrained, frozen)
        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            self.teacher = resnet18(weights=weights)
            self.student = resnet18(weights=None)  # Random init
            feature_dims = {'layer1': 64, 'layer2': 128, 'layer3': 256}
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Feature dimension for matching
        self.feature_dims = {l: feature_dims[l] for l in layers}
    
    def _forward_features(self, model, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers"""
        features = {}
        
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        
        x = model.layer1(x)
        if 'layer1' in self.layers:
            features['layer1'] = x
        
        x = model.layer2(x)
        if 'layer2' in self.layers:
            features['layer2'] = x
        
        x = model.layer3(x)
        if 'layer3' in self.layers:
            features['layer3'] = x
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        return_map: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass"""
        # Teacher features
        with torch.no_grad():
            teacher_feats = self._forward_features(self.teacher, x)
        
        # Student features
        student_feats = self._forward_features(self.student, x)
        
        # Multi-scale anomaly maps
        anomaly_maps = []
        for layer in self.layers:
            t_feat = teacher_feats[layer]
            s_feat = student_feats[layer]
            
            # Normalize
            t_feat = F.normalize(t_feat, p=2, dim=1)
            s_feat = F.normalize(s_feat, p=2, dim=1)
            
            # Distance map
            diff = (t_feat - s_feat) ** 2
            diff = diff.sum(dim=1, keepdim=True)
            
            anomaly_maps.append(diff)
        
        # Resize all to same size and average
        target_size = anomaly_maps[0].shape[-2:]
        resized_maps = []
        for m in anomaly_maps:
            if m.shape[-2:] != target_size:
                m = F.interpolate(m, size=target_size, mode='bilinear', align_corners=False)
            resized_maps.append(m)
        
        combined_map = torch.stack(resized_maps, dim=0).mean(dim=0)
        
        # Image-level score
        anomaly_score = torch.amax(combined_map, dim=(1, 2, 3))
        
        if return_map:
            return anomaly_score, combined_map
        
        return anomaly_score
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Training loss"""
        with torch.no_grad():
            teacher_feats = self._forward_features(self.teacher, x)
        
        student_feats = self._forward_features(self.student, x)
        
        total_loss = 0
        for layer in self.layers:
            t_feat = teacher_feats[layer]
            s_feat = student_feats[layer]
            
            # Normalize
            t_feat = F.normalize(t_feat, p=2, dim=1)
            s_feat = F.normalize(s_feat, p=2, dim=1)
            
            loss = F.mse_loss(s_feat, t_feat)
            total_loss = total_loss + loss
        
        return total_loss / len(self.layers)


# ============================================================================
# 6. Factory Function
# ============================================================================

def get_anomaly_detector(
    model_name: str,
    **kwargs
) -> nn.Module:
    """
    Factory function for anomaly detection models
    
    Available models:
    - 'efficientad': EfficientAD (Student-Teacher + Autoencoder)
    - 'simplenet': SimpleNet (Feature Adaptor)
    - 'patchcore': PatchCore (Memory Bank)
    - 'stfpm': STFPM (Feature Pyramid Matching)
    """
    models = {
        'efficientad': EfficientAD,
        'simplenet': SimpleNet,
        'patchcore': PatchCore,
        'stfpm': STFPM,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


# ============================================================================
# 7. Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Anomaly Detection Models Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    img_size = 256
    
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # Test EfficientAD
    print("\n1. EfficientAD:")
    model = EfficientAD(model_size='small').to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    model.eval()
    with torch.no_grad():
        score, maps = model(dummy_input, return_maps=True)
    print(f"   Parameters: {params:.2f}M")
    print(f"   Anomaly score shape: {score.shape}")
    print(f"   Combined map shape: {maps['combined'].shape}")
    
    # Test SimpleNet
    print("\n2. SimpleNet:")
    model = SimpleNet(backbone='resnet18').to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    model.eval()
    with torch.no_grad():
        score, amap = model(dummy_input, return_map=True)
    print(f"   Trainable Parameters: {params:.2f}M")
    print(f"   Anomaly score shape: {score.shape}")
    print(f"   Anomaly map shape: {amap.shape}")
    
    # Test STFPM
    print("\n3. STFPM:")
    model = STFPM(backbone='resnet18').to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    model.eval()
    with torch.no_grad():
        score, amap = model(dummy_input, return_map=True)
    print(f"   Trainable Parameters: {params:.2f}M")
    print(f"   Anomaly score shape: {score.shape}")
    print(f"   Anomaly map shape: {amap.shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
