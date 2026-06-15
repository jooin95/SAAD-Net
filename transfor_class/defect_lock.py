# -*- coding: utf-8 -*-
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


class DropPath(nn.Module):
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


class ConvFFN(nn.Module):
    def __init__(self, in_channels: int, hidden_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(in_channels * hidden_ratio)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, in_channels, 1)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SELayer(nn.Module):
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


class DefectSensitiveDynamicConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        context_channels: int,
        num_regions: int = 7,
        kernel_size: int = 7,
        num_groups: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_channels = context_channels
        self.num_regions = num_regions
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        
        self.W_q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.W_k = nn.Conv2d(context_channels + 1, in_channels, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(num_regions)
        self.W_d = nn.Linear(num_regions * num_regions, kernel_size * kernel_size)
        
        self.defect_kernel_bias = nn.Parameter(torch.zeros(1, num_groups, 1, kernel_size * kernel_size))
        self.normal_kernel_bias = nn.Parameter(torch.zeros(1, num_groups, 1, kernel_size * kernel_size))
        
        self.scale = (in_channels // num_groups) ** -0.5
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size // 2)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor, context: Tensor, anomaly_score: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        if context.shape[2:] != x.shape[2:]:
            context = F.interpolate(context, size=(H, W), mode='bilinear', align_corners=False)
        if anomaly_score.shape[2:] != x.shape[2:]:
            anomaly_score = F.interpolate(anomaly_score, size=(H, W), mode='bilinear', align_corners=False)
        
        context_with_anomaly = torch.cat([context, anomaly_score], dim=1)
        
        Q = self.W_q(x).flatten(2)
        K = self.W_k(self.pool(context_with_anomaly)).flatten(2)
        
        Q = Q.view(B, self.num_groups, self.channels_per_group, H * W)
        K = K.view(B, self.num_groups, self.channels_per_group, self.num_regions ** 2)
        
        A = torch.matmul(Q.transpose(2, 3), K) * self.scale
        D = self.W_d(A)
        
        anomaly_flat = anomaly_score.flatten(2).unsqueeze(1)
        kernel_bias = anomaly_flat.transpose(2, 3) * self.defect_kernel_bias + \
                      (1 - anomaly_flat.transpose(2, 3)) * self.normal_kernel_bias
        
        D = D + kernel_bias
        D = F.softmax(D, dim=-1)
        
        x_unfold = self.unfold(x).view(B, self.num_groups, self.channels_per_group,
                                        self.kernel_size ** 2, H * W)
        D = D.view(B, self.num_groups, H * W, self.kernel_size ** 2).transpose(2, 3)
        D = D.unsqueeze(2)
        
        out = (x_unfold * D).sum(dim=3)
        out = out.view(B, C, H, W)
        out = self.proj(out)
        
        return x + self.gamma * out


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
    def __init__(self, in_channels: int, embed_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
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


class BasicBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, drop_path: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        
        self.dwconv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(channels * 4, channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))
    
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class DynamicBlock(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        context_channels: int,
        kernel_size: int = 7,
        num_groups: int = 4,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = LayerNorm2d(feature_channels)
        self.norm2 = LayerNorm2d(feature_channels)
        
        self.ds_dconv = DefectSensitiveDynamicConv(
            feature_channels, context_channels,
            num_regions=7, kernel_size=kernel_size, num_groups=num_groups
        )
        
        self.saa = ScaleAdaptiveAttention(feature_channels, num_scales=4)
        self.ffn = ConvFFN(feature_channels, hidden_ratio=4.0)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.context_update = nn.Sequential(
            nn.Conv2d(feature_channels, context_channels, 1),
            LayerNorm2d(context_channels)
        )
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(
        self, 
        x: Tensor, 
        context: Tensor, 
        context_initial: Tensor,
        anomaly_score: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = x + self.drop_path(self.ds_dconv(self.norm1(x), context, anomaly_score))
        
        x_attn, scale_weights = self.saa(x)
        x = x + self.drop_path(x_attn - x)
        
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        context_new = self.context_update(x)
        if context_new.shape[2:] != context.shape[2:]:
            context_new = F.interpolate(context_new, size=context.shape[2:], 
                                        mode='bilinear', align_corners=False)
        context = self.alpha * context_new + self.beta * context_initial
        
        return x, context, scale_weights


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, 3, stride=2, padding=1),
            LayerNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
            LayerNorm2d(embed_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = LayerNorm2d(out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


class BaseNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dims: List[int] = [64, 128],
        depths: List[int] = [2, 2],
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(in_channels, dims[0])
        
        dp_rates = [drop_path_rate * i / sum(depths) for i in range(sum(depths))]
        self.stage1 = nn.Sequential(*[
            BasicBlock(dims[0], kernel_size=7, drop_path=dp_rates[i])
            for i in range(depths[0])
        ])
        
        self.down1 = Downsample(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[
            BasicBlock(dims[1], kernel_size=7, drop_path=dp_rates[depths[0] + i])
            for i in range(depths[1])
        ])
        
        self.out_channels = dims[1]
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.patch_embed(x)
        feat1 = self.stage1(x)
        x = self.down1(feat1)
        feat2 = self.stage2(x)
        return feat1, feat2


class AnomalyNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        self.down = Downsample(in_channels, out_channels)
        
        self.blocks = nn.Sequential(*[
            BasicBlock(out_channels, kernel_size=7, drop_path=drop_path_rate)
            for _ in range(depth)
        ])
        
        self.anomaly_gen = AnomalyScoreGenerator(out_channels, hidden_dim=out_channels // 2)
        self.out_channels = out_channels
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.down(x)
        context = self.blocks(x)
        anomaly_score, _ = self.anomaly_gen(context)
        return context, anomaly_score


class RefineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        context_channels: int,
        dims: List[int] = [256, 384],
        depths: List[int] = [6, 2],
        num_groups: List[int] = [4, 6],
        drop_path_rate: float = 0.3
    ):
        super().__init__()
        
        self.context_channels = context_channels
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 1),
            LayerNorm2d(dims[0])
        )
        
        self.down_to_context = Downsample(dims[0], dims[0])
        
        self.context_proj = nn.Sequential(
            nn.Conv2d(context_channels, dims[0] // 4, 1),
            LayerNorm2d(dims[0] // 4)
        )
        self.context_dim = dims[0] // 4
        
        total_depth = sum(depths)
        dp_rates = [drop_path_rate * i / total_depth for i in range(total_depth)]
        
        self.stage3 = nn.ModuleList([
            DynamicBlock(dims[0], self.context_dim, kernel_size=7, 
                        num_groups=num_groups[0], drop_path=dp_rates[i])
            for i in range(depths[0])
        ])
        
        self.down_to_stage4 = Downsample(dims[0], dims[1])
        self.context_down = Downsample(self.context_dim, self.context_dim)
        
        self.stage4 = nn.ModuleList([
            DynamicBlock(dims[1], self.context_dim, kernel_size=7,
                        num_groups=num_groups[1], drop_path=dp_rates[depths[0] + i])
            for i in range(depths[1])
        ])
        
        self.out_channels = dims[-1]
    
    def forward(
        self, 
        x: Tensor, 
        context: Tensor, 
        anomaly_score: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        scale_weights_list = []
        
        x = self.input_proj(x)
        x = self.down_to_context(x)
        
        context = self.context_proj(context)
        context_initial = context.clone()
        
        for block in self.stage3:
            x, context, scale_weights = block(x, context, context_initial, anomaly_score)
            scale_weights_list.append(scale_weights)
        
        x = self.down_to_stage4(x)
        context = self.context_down(context)
        context_initial_4 = self.context_down(context_initial)
        anomaly_score_4 = F.interpolate(anomaly_score, size=x.shape[2:], 
                                         mode='bilinear', align_corners=False)
        
        for block in self.stage4:
            x, context, scale_weights = block(x, context, context_initial_4, anomaly_score_4)
            scale_weights_list.append(scale_weights)
        
        return x, scale_weights_list


class DefectLoCK(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        base_dims: List[int] = [64, 128],
        base_depths: List[int] = [2, 2],
        anomaly_dim: int = 256,
        anomaly_depth: int = 2,
        refine_dims: List[int] = [256, 384],
        refine_depths: List[int] = [6, 2],
        refine_groups: List[int] = [4, 6],
        drop_path_rate: float = 0.15,
        dropout: float = 0.3,
        use_contrastive: bool = True,
        contrastive_dim: int = 128
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_contrastive = use_contrastive
        
        self.base_net = BaseNet(
            in_channels, base_dims, base_depths, 
            drop_path_rate * 0.3
        )
        
        self.anomaly_net = AnomalyNet(
            self.base_net.out_channels, anomaly_dim, 
            anomaly_depth, drop_path_rate * 0.2
        )
        
        self.refine_net = RefineNet(
            self.base_net.out_channels, anomaly_dim,
            refine_dims, refine_depths, refine_groups,
            drop_path_rate
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.refine_net.out_channels),
            nn.Dropout(dropout),
            nn.Linear(self.refine_net.out_channels, num_classes)
        )
        
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(anomaly_dim),
            nn.Dropout(dropout),
            nn.Linear(anomaly_dim, num_classes)
        )
        
        if use_contrastive:
            self.contrastive_head = ContrastiveDefectHead(
                self.refine_net.out_channels, 
                contrastive_dim
            )
        
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
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _, feat = self.base_net(x)
        context, anomaly_score = self.anomaly_net(feat)
        refined, scale_weights_list = self.refine_net(feat, context, anomaly_score)
        
        pooled = self.pool(refined).flatten(1)
        logits = self.head(pooled)
        aux_logits = self.aux_head(context)
        
        embeddings = None
        if self.use_contrastive:
            embeddings = self.contrastive_head(pooled)
        
        return {
            'logits': logits,
            'aux_logits': aux_logits,
            'embeddings': embeddings,
            'anomaly_score': anomaly_score,
            'scale_weights': scale_weights_list[-1] if scale_weights_list else None
        }
    
    def forward_simple(self, x: Tensor) -> Tensor:
        outputs = self.forward(x)
        return outputs['logits']
    
    def get_anomaly_map(self, x: Tensor) -> Tensor:
        _, feat = self.base_net(x)
        _, anomaly_score = self.anomaly_net(feat)
        return anomaly_score


class DefectLoCKLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        aux_weight: float = 0.4,
        contrastive_weight: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.aux_weight = aux_weight
        self.contrastive_weight = contrastive_weight
        
        self.main_loss = FocalLossWithSmoothing(focal_gamma, label_smoothing)
        self.aux_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.07)
    
    def forward(self, outputs: Dict[str, Tensor], targets: Tensor) -> Dict[str, Tensor]:
        losses = {}
        
        losses['main'] = self.main_loss(outputs['logits'], targets)
        losses['aux'] = self.aux_loss(outputs['aux_logits'], targets)
        
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


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                   label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_defect_lock(
    num_classes: int = 7,
    model_size: str = 't',
    use_contrastive: bool = True,
    **kwargs
) -> DefectLoCK:
    configs = {
        'xt': {
            'base_dims': [48, 96],
            'base_depths': [2, 2],
            'anomaly_dim': 192,
            'anomaly_depth': 2,
            'refine_dims': [192, 256],
            'refine_depths': [4, 2],
            'refine_groups': [3, 4],
            'drop_path_rate': 0.1,
            'dropout': 0.3
        },
        't': {
            'base_dims': [64, 128],
            'base_depths': [2, 2],
            'anomaly_dim': 256,
            'anomaly_depth': 2,
            'refine_dims': [256, 384],
            'refine_depths': [6, 2],
            'refine_groups': [4, 6],
            'drop_path_rate': 0.15,
            'dropout': 0.3
        },
        's': {
            'base_dims': [80, 160],
            'base_depths': [3, 3],
            'anomaly_dim': 320,
            'anomaly_depth': 3,
            'refine_dims': [320, 448],
            'refine_depths': [9, 3],
            'refine_groups': [5, 7],
            'drop_path_rate': 0.3,
            'dropout': 0.3
        },
        'b': {
            'base_dims': [96, 192],
            'base_depths': [4, 4],
            'anomaly_dim': 384,
            'anomaly_depth': 4,
            'refine_dims': [384, 512],
            'refine_depths': [12, 4],
            'refine_groups': [6, 8],
            'drop_path_rate': 0.4,
            'dropout': 0.3
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    cfg = configs[model_size]
    cfg['use_contrastive'] = use_contrastive
    cfg.update(kwargs)
    
    return DefectLoCK(num_classes=num_classes, **cfg)


def defect_lock_xt(num_classes: int = 7, **kwargs) -> DefectLoCK:
    return get_defect_lock(num_classes, 'xt', **kwargs)

def defect_lock_t(num_classes: int = 7, **kwargs) -> DefectLoCK:
    return get_defect_lock(num_classes, 't', **kwargs)

def defect_lock_s(num_classes: int = 7, **kwargs) -> DefectLoCK:
    return get_defect_lock(num_classes, 's', **kwargs)

def defect_lock_b(num_classes: int = 7, **kwargs) -> DefectLoCK:
    return get_defect_lock(num_classes, 'b', **kwargs)


if __name__ == '__main__':
    print("=" * 70)
    print("DefectLoCK: Defect-aware Look-Closely Network")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    for size in ['xt', 't']:
        print(f"\n{'='*50}")
        print(f"Testing DefectLoCK-{size.upper()}")
        print(f"{'='*50}")
        
        model = get_defect_lock(num_classes=7, model_size=size).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params / 1e6:.2f}M")
        
        x = torch.randn(2, 3, 512, 512).to(device)
        
        model.train()
        outputs = model(x)
        
        print(f"Input: {x.shape}")
        print(f"Logits: {outputs['logits'].shape}")
        print(f"Aux logits: {outputs['aux_logits'].shape}")
        print(f"Embeddings: {outputs['embeddings'].shape if outputs['embeddings'] is not None else 'None'}")
        print(f"Anomaly score: {outputs['anomaly_score'].shape}")
        print(f"Scale weights: {outputs['scale_weights'].shape if outputs['scale_weights'] is not None else 'None'}")
        
        loss_fn = DefectLoCKLoss(num_classes=7)
        targets = torch.randint(0, 7, (2,)).to(device)
        losses = loss_fn(outputs, targets)
        
        print(f"\nLosses:")
        for name, value in losses.items():
            print(f"  {name}: {value.item():.4f}")
        
        model.eval()
        with torch.no_grad():
            logits = model.forward_simple(x)
            anomaly_map = model.get_anomaly_map(x)
        print(f"\nInference logits: {logits.shape}")
        print(f"Anomaly map: {anomaly_map.shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)