# -*- coding: utf-8 -*-
"""
Swin V2 OverLoCK Style - Overview-first-Look-Closely-next Architecture
========================================================================

Inspired by OverLoCK (CVPR 2025), this model implements:
1. Deep-stage Decomposition Strategy (DDS)
   - Base-Net: Low/mid-level feature extraction (Stages 1-2)
   - Overview-Net: Rapid global context acquisition (lightweight)
   - Focus-Net: Fine-grained perception with top-down guidance (Stages 3-4)

2. Context-Mixing Dynamic Convolution (ContMix)
   - Token-wise global context representation via affinity matrix
   - Dynamic kernel generation from context prior
   - Preserves local inductive bias while modeling long-range dependencies

3. Gated Dynamic Spatial Aggregator (GDSA)
   - ContMix as core token mixer
   - Gated mechanism for noise elimination

4. Context Flow
   - Context prior propagation and update through Focus-Net
   - Prevents context dilution with initial prior residual

Reference: OverLoCK: An Overview-first-Look-Closely-next ConvNet (arXiv:2502.20087)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Dict
import math

try:
    from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b
    from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights
    SWIN_V2_AVAILABLE = True
except ImportError:
    SWIN_V2_AVAILABLE = False
    print("Warning: torchvision Swin V2 not available")


# ============================================================================
# Basic Building Blocks
# ============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm for 2D feature maps (B, C, H, W)"""
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


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale


class ConvFFN(nn.Module):
    """Convolutional Feed-Forward Network"""
    def __init__(self, in_channels: int, hidden_channels: int = None, out_channels: int = None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels * 4
        out_channels = out_channels or in_channels
        
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DilatedRepConv(nn.Module):
    """
    Dilated Re-parameterizable Convolution
    Combines multiple dilated convolutions for multi-scale perception
    """
    def __init__(self, channels: int, kernel_size: int = 13):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Main large kernel
        self.large_conv = nn.Conv2d(
            channels, channels, kernel_size, 
            padding=padding, groups=channels, bias=False
        )
        
        # Small kernel for local details
        self.small_conv = nn.Conv2d(
            channels, channels, 5,
            padding=2, groups=channels, bias=False
        )
        
        # Dilated conv for expanded receptive field
        self.dilated_conv = nn.Conv2d(
            channels, channels, 5,
            padding=4, dilation=2, groups=channels, bias=False
        )
        
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.large_conv(x) + self.small_conv(x) + self.dilated_conv(x)
        return self.bn(out)


# ============================================================================
# Context-Mixing Dynamic Convolution (ContMix) - Core Innovation
# ============================================================================

class ContMix(nn.Module):
    """
    Context-Mixing Dynamic Convolution (ContMix)
    
    Key Innovation from OverLoCK:
    - Computes affinity between each token and region centers in context prior
    - Uses affinity to generate spatially-varying dynamic kernels
    - Each kernel weight carries global information from top-down context
    - Captures long-range dependencies while preserving local inductive bias
    
    Args:
        in_channels: Input feature channels (Z_i)
        context_channels: Context prior channels (P_i)
        num_regions: Number of region centers (S x S), default 7
        kernel_size: Dynamic kernel size (K x K), default 13 for large, 5 for small
        num_groups: Number of groups for grouped convolution
    """
    def __init__(
        self,
        in_channels: int,
        context_channels: int,
        num_regions: int = 7,
        kernel_size: int = 13,
        num_groups: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.num_regions = num_regions
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        
        # Q projection from input feature Z_i
        self.W_q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # K projection from context prior P_i
        self.W_k = nn.Conv2d(context_channels, in_channels, 1, bias=False)
        
        # Adaptive pooling to get S x S region centers
        self.pool = nn.AdaptiveAvgPool2d(num_regions)
        
        # Learnable linear layer to transform affinity to kernel weights
        # W_d in paper: R^{S^2} -> R^{K^2}
        self.W_d = nn.Linear(num_regions * num_regions, kernel_size * kernel_size)
        
        # Temperature for softmax
        self.scale = (in_channels // num_groups) ** -0.5
        
        # Unfold for efficient convolution
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size // 2)
        
        # Dilated RepConv for channel diversity
        self.repconv = DilatedRepConv(in_channels, kernel_size)
    
    def forward(self, x: Tensor, context_prior: Tensor) -> Tensor:
        """
        Args:
            x: Input feature map (B, C_in, H, W) - corresponds to Z_i
            context_prior: Context prior (B, C_p, H, W) - corresponds to P_i
        
        Returns:
            Output feature map (B, C_in, H, W)
        """
        B, C, H, W = x.shape
        
        # Step 1: Compute Q from input feature
        Q = self.W_q(x)  # (B, C, H, W)
        Q = Q.flatten(2)  # (B, C, HW)
        
        # Step 2: Compute K from context prior via region centers
        K = self.W_k(self.pool(context_prior))  # (B, C, S, S)
        K = K.flatten(2)  # (B, C, S^2)
        
        # Step 3: Split into groups (like multi-head attention)
        Q = Q.view(B, self.num_groups, self.channels_per_group, H * W)  # (B, G, C/G, HW)
        K = K.view(B, self.num_groups, self.channels_per_group, self.num_regions ** 2)  # (B, G, C/G, S^2)
        
        # Step 4: Compute affinity matrix A = Q^T @ K for each group
        # Q^T: (B, G, HW, C/G), K: (B, G, C/G, S^2)
        # A: (B, G, HW, S^2)
        A = torch.matmul(Q.transpose(2, 3), K) * self.scale
        
        # Step 5: Transform affinity to dynamic kernel weights
        # D = softmax(A @ W_d)
        D = self.W_d(A)  # (B, G, HW, K^2)
        D = F.softmax(D, dim=-1)  # Normalize
        
        # Step 6: Reshape D to kernel shape
        # D: (B, G, H, W, K, K)
        D = D.view(B, self.num_groups, H, W, self.kernel_size, self.kernel_size)
        
        # Step 7: Apply dynamic convolution via unfold-multiply-sum
        # Unfold input: (B, C*K*K, H*W)
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(B, self.num_groups, self.channels_per_group, 
                                  self.kernel_size * self.kernel_size, H * W)
        # (B, G, C/G, K^2, HW)
        
        # Reshape D for multiplication: (B, G, 1, K^2, HW)
        D_flat = D.view(B, self.num_groups, 1, self.kernel_size * self.kernel_size, H * W)
        
        # Weighted sum: (B, G, C/G, HW)
        out = (x_unfold * D_flat).sum(dim=3)
        
        # Reshape back: (B, C, H, W)
        out = out.view(B, C, H, W)
        
        # Add RepConv for channel diversity
        out = out + self.repconv(x)
        
        return out


class ContMixMultiScale(nn.Module):
    """
    Multi-scale ContMix combining large and small kernels
    Following OverLoCK's design: half groups for large kernels, half for small
    """
    def __init__(
        self,
        in_channels: int,
        context_channels: int,
        num_regions: int = 7,
        large_kernel: int = 13,
        small_kernel: int = 5,
        num_groups: int = 4
    ):
        super().__init__()
        assert num_groups >= 2, "Need at least 2 groups for multi-scale"
        
        large_groups = num_groups // 2
        small_groups = num_groups - large_groups
        
        large_channels = in_channels * large_groups // num_groups
        small_channels = in_channels - large_channels
        
        self.large_channels = large_channels
        self.small_channels = small_channels
        
        self.contmix_large = ContMix(
            large_channels, context_channels, 
            num_regions, large_kernel, large_groups
        )
        self.contmix_small = ContMix(
            small_channels, context_channels,
            num_regions, small_kernel, small_groups
        )
        
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x: Tensor, context_prior: Tensor) -> Tensor:
        x_large = x[:, :self.large_channels]
        x_small = x[:, self.large_channels:]
        
        out_large = self.contmix_large(x_large, context_prior)
        out_small = self.contmix_small(x_small, context_prior)
        
        out = torch.cat([out_large, out_small], dim=1)
        return self.fusion(out)


# ============================================================================
# Gated Dynamic Spatial Aggregator (GDSA)
# ============================================================================

class GDSA(nn.Module):
    """
    Gated Dynamic Spatial Aggregator
    
    Uses ContMix as core token mixer with gated mechanism for noise elimination
    
    Structure:
    - Input: Concatenated [Z_i, P_i]
    - Branch 1: ContMix for dynamic spatial aggregation
    - Branch 2: Gate generation via 1x1 Conv + SiLU
    - Output: Element-wise multiplication of branches
    """
    def __init__(
        self,
        feature_channels: int,
        context_channels: int,
        num_regions: int = 7,
        large_kernel: int = 13,
        small_kernel: int = 5,
        num_groups: int = 4
    ):
        super().__init__()
        total_channels = feature_channels + context_channels
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(total_channels, feature_channels, 1),
            nn.SiLU(inplace=True)
        )
        
        # ContMix branch
        self.contmix = ContMixMultiScale(
            feature_channels, context_channels,
            num_regions, large_kernel, small_kernel, num_groups
        )
        
        # SE Layer after ContMix
        self.se = SELayer(feature_channels)
        
        # Gate branch
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, feature_channels, 1),
            nn.SiLU(inplace=True)
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(feature_channels, feature_channels, 1)
    
    def forward(self, z: Tensor, p: Tensor) -> Tensor:
        """
        Args:
            z: Feature map (B, C_z, H, W)
            p: Context prior (B, C_p, H, W)
        
        Returns:
            Output feature (B, C_z, H, W)
        """
        # Concatenate feature and context
        x = torch.cat([z, p], dim=1)
        
        # ContMix branch
        feat = self.input_proj(x)
        feat = self.contmix(feat, p)
        feat = self.se(feat)
        
        # Gate branch
        gate = self.gate(x)
        
        # Gated output
        out = feat * gate
        out = self.output_proj(out)
        
        return out


# ============================================================================
# Network Building Blocks
# ============================================================================

class BasicBlock(nn.Module):
    """
    Basic Block for Base-Net and Overview-Net
    
    Structure: 3x3 DWConv -> Norm -> Dilated RepConv -> SE -> ConvFFN
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 13,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # Local perception
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # Main branch
        self.norm1 = LayerNorm2d(channels)
        self.repconv = DilatedRepConv(channels, kernel_size)
        self.se = SELayer(channels)
        
        # FFN
        self.norm2 = LayerNorm2d(channels)
        self.ffn = ConvFFN(channels)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        # Local perception with residual
        x = x + self.dwconv(x)
        
        # Main branch
        shortcut = x
        x = self.norm1(x)
        x = self.repconv(x)
        x = self.se(x)
        x = shortcut + self.drop_path(x)
        
        # FFN
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = shortcut + self.drop_path(x)
        
        return x


class DynamicBlock(nn.Module):
    """
    Dynamic Block for Focus-Net
    
    Structure:
    - Input: Z_i (feature) and P_i (context prior)
    - 3x3 DWConv for local perception
    - GDSA for dynamic spatial aggregation with top-down guidance
    - ConvFFN
    - Output: Updated Z_{i+1} and P'_i
    
    Context Flow:
    - P_{i+1} = alpha * P'_i + beta * P_0 (prevents dilution)
    """
    def __init__(
        self,
        feature_channels: int,
        context_channels: int,
        kernel_size: int = 13,
        num_regions: int = 7,
        num_groups: int = 4,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.feature_channels = feature_channels
        self.context_channels = context_channels
        
        # Local perception
        self.dwconv = nn.Conv2d(feature_channels + context_channels, 
                                feature_channels + context_channels, 
                                3, padding=1, 
                                groups=feature_channels + context_channels)
        
        # GDSA
        self.norm1 = LayerNorm2d(feature_channels + context_channels)
        self.gdsa = GDSA(
            feature_channels, context_channels,
            num_regions, kernel_size, 5, num_groups
        )
        
        # FFN
        self.norm2 = LayerNorm2d(feature_channels)
        self.ffn = ConvFFN(feature_channels)
        
        # Context prior update parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(
        self, 
        z: Tensor, 
        p: Tensor, 
        p_initial: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: Feature map (B, C_z, H, W)
            p: Current context prior (B, C_p, H, W)
            p_initial: Initial context prior (B, C_p, H, W)
        
        Returns:
            z_out: Updated feature (B, C_z, H, W)
            p_out: Updated context prior (B, C_p, H, W)
        """
        # Concatenate for local perception
        x = torch.cat([z, p], dim=1)
        x = x + self.dwconv(x)
        
        # Split back
        z = x[:, :self.feature_channels]
        p = x[:, self.feature_channels:]
        
        # Normalize
        x_norm = self.norm1(torch.cat([z, p], dim=1))
        z_norm = x_norm[:, :self.feature_channels]
        p_norm = x_norm[:, self.feature_channels:]
        
        # GDSA with residual
        z = z + self.drop_path(self.gdsa(z_norm, p_norm))
        
        # FFN
        z = z + self.drop_path(self.ffn(self.norm2(z)))
        
        # Update context prior with dilution prevention
        # P_{i+1} = alpha * P'_i + beta * P_0
        p_out = self.alpha * p + self.beta * p_initial
        
        return z, p_out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================================
# Embedding Layers
# ============================================================================

class PatchEmbed(nn.Module):
    """Patch Embedding with Conv"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            LayerNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            LayerNorm2d(out_channels)
        ) if stride == 4 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            LayerNorm2d(out_channels)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Downsample layer"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = LayerNorm2d(out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


# ============================================================================
# Sub-Networks: Base-Net, Overview-Net, Focus-Net
# ============================================================================

class BaseNet(nn.Module):
    """
    Base-Net: Encodes low/mid-level features (Stages 1-2)
    
    Output: Mid-level feature map at H/8 x W/8 resolution
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [64, 128],
        blocks: List[int] = [2, 2],
        kernel_sizes: List[int] = [7, 7],
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        # Patch embedding: H/4 x W/4
        self.patch_embed = PatchEmbed(in_channels, channels[0], stride=4)
        
        # Stage 1
        self.stage1 = nn.Sequential(*[
            BasicBlock(channels[0], kernel_sizes[0], 
                       drop_path=drop_path_rate * i / sum(blocks))
            for i in range(blocks[0])
        ])
        
        # Downsample: H/8 x W/8
        self.down1 = Downsample(channels[0], channels[1])
        
        # Stage 2
        dp_start = blocks[0]
        self.stage2 = nn.Sequential(*[
            BasicBlock(channels[1], kernel_sizes[1],
                       drop_path=drop_path_rate * (dp_start + i) / sum(blocks))
            for i in range(blocks[1])
        ])
        
        self.out_channels = channels[1]
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            x1: Stage 1 output (H/4 x W/4) for FPN
            x2: Stage 2 output (H/8 x W/8) - main output
        """
        x = self.patch_embed(x)
        x1 = self.stage1(x)
        x = self.down1(x1)
        x2 = self.stage2(x)
        return x1, x2


class OverviewNet(nn.Module):
    """
    Overview-Net: Rapidly acquires global context ("overview first")
    
    - Lightweight design
    - Immediately downsamples to H/16 x W/16
    - Produces semantically meaningful but coarse context
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int = 2,
        kernel_size: int = 7,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        # Downsample to H/16
        self.downsample = Downsample(in_channels, out_channels)
        
        # Lightweight blocks
        self.blocks = nn.Sequential(*[
            BasicBlock(out_channels, kernel_size, drop_path=drop_path_rate)
            for _ in range(blocks)
        ])
        
        self.out_channels = out_channels
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Mid-level feature from Base-Net (B, C, H/8, W/8)
        
        Returns:
            Context feature (B, C_out, H/16, W/16)
        """
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class FocusNet(nn.Module):
    """
    Focus-Net: Fine-grained perception with top-down guidance ("look closely next")
    
    - Deep and powerful
    - Receives context prior from Overview-Net
    - Uses Dynamic Blocks with ContMix
    - Context prior is updated through the network
    """
    def __init__(
        self,
        in_channels: int,
        context_channels: int,
        channels: List[int] = [256, 512],
        blocks: List[int] = [6, 2],
        kernel_sizes: List[int] = [13, 7],
        num_groups: List[int] = [4, 8],
        num_regions: int = 7,
        drop_path_rate: float = 0.3
    ):
        super().__init__()
        
        # Context prior projection (channel reduction)
        self.context_proj = nn.Sequential(
            nn.Conv2d(context_channels, channels[0] // 4, 1),
            LayerNorm2d(channels[0] // 4)
        )
        self.context_channels = channels[0] // 4
        
        # Input projection for feature
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 1),
            LayerNorm2d(channels[0])
        )
        
        # Stage 3 (H/8 resolution, but we work at H/16 with context)
        # First downsample feature to match context
        self.down_to_stage3 = Downsample(channels[0], channels[0])
        
        # Context upsampling for Stage 3 (H/16 -> H/16, same resolution)
        self.context_up_stage3 = nn.Identity()  # Same resolution
        
        total_blocks = sum(blocks)
        dp_idx = 0
        
        self.stage3_blocks = nn.ModuleList([
            DynamicBlock(
                channels[0], self.context_channels,
                kernel_sizes[0], num_regions, num_groups[0],
                drop_path=drop_path_rate * dp_idx / total_blocks
            )
            for dp_idx in range(blocks[0])
        ])
        
        # Downsample to Stage 4: H/32
        self.down_to_stage4 = Downsample(channels[0], channels[1])
        
        # Context for Stage 4 (downsample)
        self.context_down_stage4 = Downsample(self.context_channels, self.context_channels)
        
        self.stage4_blocks = nn.ModuleList([
            DynamicBlock(
                channels[1], self.context_channels,
                kernel_sizes[1], num_regions, num_groups[1],
                drop_path=drop_path_rate * (blocks[0] + dp_idx) / total_blocks
            )
            for dp_idx in range(blocks[1])
        ])
        
        self.channels = channels
    
    def forward(
        self, 
        x: Tensor, 
        context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: Mid-level feature from Base-Net (B, C, H/8, W/8)
            context: Context from Overview-Net (B, C_ctx, H/16, W/16)
        
        Returns:
            out3: Stage 3 output (B, C3, H/16, W/16)
            out4: Stage 4 output (B, C4, H/32, W/32)
            context_final: Final context prior
        """
        # Project input and context
        z = self.input_proj(x)
        z = self.down_to_stage3(z)  # H/8 -> H/16
        
        p = self.context_proj(context)
        p_initial = p.clone()
        
        # Stage 3
        for block in self.stage3_blocks:
            z, p = block(z, p, p_initial)
        out3 = z
        
        # Downsample to Stage 4
        z = self.down_to_stage4(z)
        p = self.context_down_stage4(p)
        p_initial_4 = self.context_down_stage4(p_initial)
        
        # Stage 4
        for block in self.stage4_blocks:
            z, p = block(z, p, p_initial_4)
        out4 = z
        
        return out3, out4, p


# ============================================================================
# Main Model: SwinV2OverLoCK
# ============================================================================

class SwinV2OverLoCK(nn.Module):
    """
    Swin V2 with OverLoCK-style Architecture
    
    Overview-first-Look-Closely-next design:
    1. Base-Net extracts low/mid-level features
    2. Overview-Net rapidly acquires global context
    3. Focus-Net performs fine-grained perception with top-down guidance
    
    Features:
    - Context-Mixing Dynamic Convolution (ContMix)
    - Gated Dynamic Spatial Aggregator (GDSA)
    - Context Flow with dilution prevention
    - Auxiliary loss on Overview-Net during training
    """
    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        # Base-Net config
        base_channels: List[int] = [64, 128],
        base_blocks: List[int] = [2, 2],
        base_kernels: List[int] = [7, 7],
        # Overview-Net config  
        overview_channels: int = 256,
        overview_blocks: int = 2,
        overview_kernel: int = 7,
        # Focus-Net config
        focus_channels: List[int] = [256, 512],
        focus_blocks: List[int] = [6, 2],
        focus_kernels: List[int] = [13, 7],
        focus_groups: List[int] = [4, 8],
        num_regions: int = 7,
        # Training config
        drop_path_rate: float = 0.15,
        use_aux_loss: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_aux_loss = use_aux_loss
        
        # Sub-networks
        self.base_net = BaseNet(
            in_channels, base_channels, base_blocks, base_kernels, 
            drop_path_rate * 0.3
        )
        
        self.overview_net = OverviewNet(
            self.base_net.out_channels, overview_channels,
            overview_blocks, overview_kernel, drop_path_rate * 0.2
        )
        
        self.focus_net = FocusNet(
            self.base_net.out_channels, overview_channels,
            focus_channels, focus_blocks, focus_kernels, focus_groups,
            num_regions, drop_path_rate
        )
        
        # Classification heads
        # Main head on Focus-Net output
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            LayerNorm1d(focus_channels[-1]),
            nn.Dropout(0.1),
            nn.Linear(focus_channels[-1], num_classes)
        )
        
        # Auxiliary head on Overview-Net output (for training)
        if use_aux_loss:
            self.aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                LayerNorm1d(overview_channels),
                nn.Dropout(0.1),
                nn.Linear(overview_channels, num_classes)
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            During training with aux_loss: (main_logits, aux_logits)
            During inference: main_logits
        """
        # Base-Net: Extract low/mid-level features
        feat1, feat2 = self.base_net(x)  # feat1: H/4, feat2: H/8
        
        # Overview-Net: Get global context ("overview first")
        context = self.overview_net(feat2)  # H/16
        
        # Focus-Net: Fine-grained perception ("look closely next")
        out3, out4, _ = self.focus_net(feat2, context)  # out3: H/16, out4: H/32
        
        # Main classification
        logits = self.head(out4)
        
        # Auxiliary loss during training
        if self.training and self.use_aux_loss:
            aux_logits = self.aux_head(context)
            return logits, aux_logits
        
        return logits
    
    def get_feature_maps(self, x: Tensor) -> Dict[str, Tensor]:
        """Get intermediate feature maps for visualization"""
        feat1, feat2 = self.base_net(x)
        context = self.overview_net(feat2)
        out3, out4, context_final = self.focus_net(feat2, context)
        
        return {
            'base_stage1': feat1,
            'base_stage2': feat2,
            'overview': context,
            'focus_stage3': out3,
            'focus_stage4': out4,
            'context_final': context_final
        }


class LayerNorm1d(nn.Module):
    """LayerNorm for 1D features"""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.ln(x)


# ============================================================================
# Loss Functions
# ============================================================================

class OverLoCKLoss(nn.Module):
    """
    Loss function for OverLoCK-style training
    
    L = L_main + aux_weight * L_aux
    
    Where:
    - L_main: Main classification loss from Focus-Net
    - L_aux: Auxiliary loss from Overview-Net (helps learn semantic context)
    """
    def __init__(
        self,
        num_classes: int,
        aux_weight: float = 0.4,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.main_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.aux_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self, 
        outputs: Tuple[Tensor, Tensor], 
        targets: Tensor
    ) -> Tensor:
        """
        Args:
            outputs: (main_logits, aux_logits) from model
            targets: Ground truth labels
        
        Returns:
            Combined loss
        """
        if isinstance(outputs, tuple):
            main_logits, aux_logits = outputs
            loss = self.main_loss(main_logits, targets)
            loss += self.aux_weight * self.aux_loss(aux_logits, targets)
        else:
            loss = self.main_loss(outputs, targets)
        
        return loss


# ============================================================================
# Model Variants Factory
# ============================================================================

def get_swin_v2_overlock(
    num_classes: int = 7,
    model_size: str = 't',
    drop_path_rate: float = None,
    use_aux_loss: bool = True
) -> SwinV2OverLoCK:
    """
    Factory function for SwinV2OverLoCK variants
    
    Args:
        num_classes: Number of output classes
        model_size: 'xt' (extreme tiny), 't' (tiny), 's' (small), 'b' (base)
        drop_path_rate: Stochastic depth rate (None for default)
        use_aux_loss: Whether to use auxiliary loss on Overview-Net
    
    Returns:
        SwinV2OverLoCK model
    """
    configs = {
        'xt': {  # Extreme Tiny (~16M params)
            'base_channels': [56, 112],
            'base_blocks': [2, 2],
            'base_kernels': [7, 7],
            'overview_channels': 256,
            'overview_blocks': 2,
            'overview_kernel': 7,
            'focus_channels': [256, 336],
            'focus_blocks': [6, 2],
            'focus_kernels': [13, 7],
            'focus_groups': [4, 6],
            'drop_path_rate': 0.1
        },
        't': {  # Tiny (~33M params)
            'base_channels': [64, 128],
            'base_blocks': [4, 4],
            'base_kernels': [7, 7],
            'overview_channels': 512,
            'overview_blocks': 2,
            'overview_kernel': 7,
            'focus_channels': [256, 512],
            'focus_blocks': [12, 2],
            'focus_kernels': [13, 7],
            'focus_groups': [4, 8],
            'drop_path_rate': 0.15
        },
        's': {  # Small (~56M params)
            'base_channels': [64, 128],
            'base_blocks': [6, 6],
            'base_kernels': [7, 7],
            'overview_channels': 512,
            'overview_blocks': 3,
            'overview_kernel': 7,
            'focus_channels': [320, 512],
            'focus_blocks': [16, 3],
            'focus_kernels': [13, 7],
            'focus_groups': [5, 8],
            'drop_path_rate': 0.4
        },
        'b': {  # Base (~95M params)
            'base_channels': [80, 160],
            'base_blocks': [8, 8],
            'base_kernels': [7, 7],
            'overview_channels': 576,
            'overview_blocks': 4,
            'overview_kernel': 7,
            'focus_channels': [384, 576],
            'focus_blocks': [20, 4],
            'focus_kernels': [13, 7],
            'focus_groups': [6, 9],
            'drop_path_rate': 0.5
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    cfg = configs[model_size]
    
    if drop_path_rate is not None:
        cfg['drop_path_rate'] = drop_path_rate
    
    return SwinV2OverLoCK(
        num_classes=num_classes,
        use_aux_loss=use_aux_loss,
        **cfg
    )


# Aliases
def swin_v2_overlock_xt(num_classes: int = 7, **kwargs) -> SwinV2OverLoCK:
    return get_swin_v2_overlock(num_classes, 'xt', **kwargs)

def swin_v2_overlock_t(num_classes: int = 7, **kwargs) -> SwinV2OverLoCK:
    return get_swin_v2_overlock(num_classes, 't', **kwargs)

def swin_v2_overlock_s(num_classes: int = 7, **kwargs) -> SwinV2OverLoCK:
    return get_swin_v2_overlock(num_classes, 's', **kwargs)

def swin_v2_overlock_b(num_classes: int = 7, **kwargs) -> SwinV2OverLoCK:
    return get_swin_v2_overlock(num_classes, 'b', **kwargs)


# ============================================================================
# Test Code
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing SwinV2OverLoCK Models")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test input
    x = torch.randn(2, 3, 512, 512).to(device)
    
    # Test each variant
    for size in ['xt', 't']:
        print(f"\n{'='*50}")
        print(f"Testing OverLoCK-{size.upper()}")
        print(f"{'='*50}")
        
        model = get_swin_v2_overlock(num_classes=7, model_size=size).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params / 1e6:.2f}M")
        print(f"Trainable params: {trainable_params / 1e6:.2f}M")
        
        # Training mode (returns aux output)
        model.train()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(x)
        
        if isinstance(outputs, tuple):
            main_out, aux_out = outputs
            print(f"Training - Main output: {main_out.shape}, Aux output: {aux_out.shape}")
        else:
            print(f"Training - Output: {outputs.shape}")
        
        # Inference mode
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        print(f"Inference - Output: {outputs.shape}")
        
        # Test loss function
        loss_fn = OverLoCKLoss(num_classes=7)
        model.train()
        outputs = model(x)
        targets = torch.randint(0, 7, (2,)).to(device)
        loss = loss_fn(outputs, targets)
        print(f"Loss: {loss.item():.4f}")
        
        # Get feature maps
        model.eval()
        with torch.no_grad():
            feat_maps = model.get_feature_maps(x)
        print("Feature maps:")
        for name, feat in feat_maps.items():
            print(f"  {name}: {feat.shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
