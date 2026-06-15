#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GC10-DET Benchmark Script
=========================
Multi-model comparison for steel surface defect classification.

Comparison Models (from train_unified.py):
  CNN Baselines:
    - resnet50: ResNet-50 (Baseline CNN)
    - resnet101: ResNet-101
    - efficientnet_b4: EfficientNet-B4
    - efficientnet_v2_s: EfficientNet V2-S
    - convnext_tiny: ConvNeXt-Tiny
    - convnext_small: ConvNeXt-Small
    - convnext_base: ConvNeXt-Base

  Transformers:
    - swin_v2_t: Swin Transformer V2-Tiny
    - swin_v2_s: Swin Transformer V2-Small
    - swin_v2_b: Swin Transformer V2-Base
    - vit_b_16: Vision Transformer B/16
    - vit_l_16: Vision Transformer L/16
    - maxvit_t: MaxViT-Tiny

  Proposed:
    - saad_net: SAAD-Net (DefectLoCK V2 Improved)

Features:
  - Multi-GPU DDP training with NCCL
  - Early stopping with rank broadcast (prevents NCCL timeout)
  - Warmup + Cosine LR scheduler
  - Mixed precision training (AMP)
  - Proper process cleanup on exit/interrupt

Usage:
    # All models
    torchrun --nproc_per_node=4 gc10_benchmark.py \\
        --data_dir /path/to/gc10/classification \\
        --output_dir ./gc10_results

    # Specific models only
    torchrun --nproc_per_node=4 gc10_benchmark.py \\
        --data_dir /path/to/gc10/classification \\
        --models swin_v2_t vit_b_16 saad_net

    # Single GPU
    python gc10_benchmark.py --data_dir /path/to/gc10/classification
"""

import os

# =============================================================================
# NCCL Environment Setup (MUST be before torch import)
# =============================================================================
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import warnings
import signal
import sys
import random
import math
import atexit
import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

warnings.filterwarnings('ignore')


# =============================================================================
# Import SAAD-Net (DefectLoCK V2 Improved)
# =============================================================================
try:
    from defect_lock_v2_improved import (
        get_defect_lock_v2_improved_for_training,
        DefectLoCKv2ImprovedLoss
    )
    SAAD_NET_AVAILABLE = True
except ImportError as e:
    SAAD_NET_AVAILABLE = False
    print(f"[WARNING] defect_lock_v2_improved.py not found: {e}")

# =============================================================================
# Import timm for flexible ViT
# =============================================================================
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not available. ViT models will be limited.")

# =============================================================================
# Import ultralytics for YOLOv8-cls
# =============================================================================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not available. YOLOv8-cls models will be skipped.")


# =============================================================================
# Global State for Cleanup
# =============================================================================
_CLEANUP_DONE = False


def _force_cleanup():
    """Force cleanup of distributed resources - safe to call multiple times"""
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    
    try:
        if dist.is_initialized():
            # Give processes time to sync
            try:
                dist.barrier(timeout=timedelta(seconds=5))
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except Exception:
        pass
    
    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Register cleanup at exit
atexit.register(_force_cleanup)


# =============================================================================
# Distributed Utilities
# =============================================================================

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training"""
    if dist.is_initialized():
        return
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30)
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    _force_cleanup()


def is_main_process() -> bool:
    """Check if current process is rank 0"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current rank"""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get world size"""
    return dist.get_world_size() if dist.is_initialized() else 1


def print_rank0(*args, **kwargs):
    """Print only on rank 0"""
    if is_main_process():
        print(*args, **kwargs)


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank to all ranks"""
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


# =============================================================================
# Dataset
# =============================================================================

class GC10Dataset(Dataset):
    """
    GC10-DET Dataset for classification.
    Supports pre-split train/val/test folder structure.
    """

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split

        # Check for pre-split structure
        split_dir = self.root_dir / split
        self.data_dir = split_dir if split_dir.exists() else self.root_dir

        # Auto-detect classes (exclude split folders)
        self.classes = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and d.name not in ['train', 'val', 'test']
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect samples
        self.samples = []
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in extensions:
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))

        if is_main_process():
            print(f"[{split.upper()}] {len(self.samples)} samples, {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing Scheduler (epoch-based)
    
    Schedule:
      - Warmup phase: Linear from 1e-7 to base_lr
      - Decay phase: Cosine from base_lr to min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = 1e-7 + (self.base_lr - 1e-7) * (epoch / max(1, self.warmup_epochs))
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# =============================================================================
# Model Builders
# =============================================================================

# List of all available models
AVAILABLE_MODELS = [
    # CNN Baselines
    'resnet50', 'resnet101',
    'efficientnet_b4', 'efficientnet_v2_s',
    'convnext_tiny', 'convnext_small', 'convnext_base',
    # Transformers
    'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
    'vit_b_16', 'vit_l_16',
    'maxvit_t',
    # YOLOv8-cls
    'yolov8_cls_s', 'yolov8_cls_m', 'yolov8_cls_l',
    # Proposed
    'saad_net',           # Full model (with CDL)
    'saad_net_no_cdl',    # Without CDL (attention_only config)
]

# Default comparison set (from paper table)
DEFAULT_BENCHMARK_MODELS = [
    'yolov8_cls_s',       # YOLOv8-cls-s (CSPDarknet) ~5.1M
    'yolov8_cls_m',       # YOLOv8-cls-m (CSPDarknet) ~15.8M
    'efficientnet_b4',    # EfficientNet-B4 ~17.6M
    'swin_v2_t',          # Swin V2-T (Baseline) ~28.3M
    'vit_b_16',           # ViT-Base/16 ~86.6M
    'saad_net_no_cdl',    # SAAD-Net w/o CDL ~29.1M
    'saad_net',           # SAAD-Net (Full)
]


class YOLOv8ClassifierWrapper(nn.Module):
    """
    Wrapper for YOLOv8 classification model to work with standard PyTorch training.
    
    YOLOv8-cls uses CSPDarknet backbone with classification head.
    This wrapper extracts the backbone and classifier for standard training.
    """
    
    def __init__(self, variant: str, num_classes: int, img_size: int = 224):
        super().__init__()
        
        # Load pretrained YOLOv8-cls model
        yolo_model = YOLO(variant)
        
        # Extract the PyTorch model
        self.model = yolo_model.model
        
        # Replace the classification head
        # YOLOv8-cls structure: model.model[-1] is the Classify head
        if hasattr(self.model, 'model') and len(self.model.model) > 0:
            classify_layer = self.model.model[-1]
            if hasattr(classify_layer, 'linear'):
                in_features = classify_layer.linear.in_features
                classify_layer.linear = nn.Linear(in_features, num_classes)
            elif hasattr(classify_layer, 'fc'):
                in_features = classify_layer.fc.in_features
                classify_layer.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits"""
        # YOLOv8 model forward
        out = self.model(x)
        
        # Handle different output formats
        if isinstance(out, (list, tuple)):
            out = out[0]
        
        return out


def build_model(model_name: str, num_classes: int, img_size: int = 224) -> nn.Module:
    """
    Build classification model by name.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        img_size: Input image size (needed for ViT and YOLOv8)
    
    Returns:
        nn.Module: The model
    """

    # ==========================================================================
    # SAAD-Net (Proposed Model) - Full version with CDL
    # ==========================================================================
    if model_name in ['saad_net', 'saad', 'defect_lock', 'defect_lock_v2_imp']:
        if not SAAD_NET_AVAILABLE:
            raise ImportError("defect_lock_v2_improved.py not found")
        return get_defect_lock_v2_improved_for_training(
            num_classes=num_classes,
            config='default'  # Full model with CDL
        )

    # ==========================================================================
    # SAAD-Net without CDL (attention_only config)
    # ==========================================================================
    if model_name == 'saad_net_no_cdl':
        if not SAAD_NET_AVAILABLE:
            raise ImportError("defect_lock_v2_improved.py not found")
        return get_defect_lock_v2_improved_for_training(
            num_classes=num_classes,
            config='attention_only'  # CBAM + LAA + S-SAA, no CDL
        )

    # ==========================================================================
    # YOLOv8-cls Models (using ultralytics)
    # ==========================================================================
    if model_name.startswith('yolov8_cls'):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        # Map model name to YOLO variant
        yolo_variants = {
            'yolov8_cls_n': 'yolov8n-cls.pt',
            'yolov8_cls_s': 'yolov8s-cls.pt',
            'yolov8_cls_m': 'yolov8m-cls.pt',
            'yolov8_cls_l': 'yolov8l-cls.pt',
            'yolov8_cls_x': 'yolov8x-cls.pt',
        }
        
        variant = yolo_variants.get(model_name, 'yolov8s-cls.pt')
        
        # Create YOLOv8 classifier wrapper
        return YOLOv8ClassifierWrapper(variant, num_classes, img_size)

    # ==========================================================================
    # ResNet Family
    # ==========================================================================
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ==========================================================================
    # EfficientNet Family
    # ==========================================================================
    if model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    # ==========================================================================
    # ConvNeXt Family
    # ==========================================================================
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_name == 'convnext_small':
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_name == 'convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    # ==========================================================================
    # Swin Transformer V2 Family
    # ==========================================================================
    if model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if model_name == 'swin_v2_s':
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if model_name == 'swin_v2_b':
        model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    # ==========================================================================
    # Vision Transformer (ViT) - Using timm for flexible image sizes
    # ==========================================================================
    if model_name == 'vit_b_16':
        if TIMM_AVAILABLE:
            # timm ViT supports any image size
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=num_classes,
                img_size=img_size
            )
            return model
        else:
            # Fallback to torchvision (only works with 224x224)
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model

    if model_name == 'vit_l_16':
        if TIMM_AVAILABLE:
            model = timm.create_model(
                'vit_large_patch16_224',
                pretrained=True,
                num_classes=num_classes,
                img_size=img_size
            )
            return model
        else:
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
        return model

    # ==========================================================================
    # MaxViT
    # ==========================================================================
    if model_name == 'maxvit_t':
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available: {', '.join(AVAILABLE_MODELS)}"
    )


def get_criterion(model_name: str, num_classes: int) -> nn.Module:
    """Get loss function for the model"""
    if model_name in ['saad_net', 'saad', 'defect_lock', 'defect_lock_v2_imp']:
        if SAAD_NET_AVAILABLE:
            return DefectLoCKv2ImprovedLoss(
                num_classes=num_classes,
                focal_gamma=2.0,
                label_smoothing=0.1,
                contrastive_weight=0.1
            )
    return nn.CrossEntropyLoss(label_smoothing=0.1)


def count_parameters(model: nn.Module) -> float:
    """Count model parameters in millions"""
    return sum(p.numel() for p in model.parameters()) / 1e6


# =============================================================================
# Data Transforms
# =============================================================================

def get_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    """Get data augmentation transforms"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    model_name: str,
    warmup_epochs: int = 5
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Progressive contrastive weight for SAAD-Net
    is_saad = model_name in ['saad_net', 'saad', 'defect_lock', 'defect_lock_v2_imp']
    contrastive_weight = 0.1 * min(1.0, epoch / max(1, warmup_epochs)) if is_saad else None

    pbar = tqdm(loader, desc=f"Train E{epoch}", leave=False) if is_main_process() else loader

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)

            # Handle different output types
            if isinstance(outputs, dict):
                logits = outputs['logits']
                if hasattr(criterion, 'forward') and 'DefectLoCK' in criterion.__class__.__name__:
                    loss_dict = criterion(outputs, labels, contrastive_weight=contrastive_weight)
                    loss = loss_dict['total']
                else:
                    loss = criterion(logits, labels)
            else:
                logits = outputs
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)

        # Handle different output types
        if isinstance(outputs, dict):
            logits = outputs['logits']
            if hasattr(criterion, 'forward') and 'DefectLoCK' in criterion.__class__.__name__:
                loss = criterion(outputs, labels)['total']
            else:
                loss = criterion(logits, labels)
        else:
            logits = outputs
            loss = criterion(logits, labels)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


# =============================================================================
# Single Model Training
# =============================================================================

def train_single_model(
    model_name: str,
    args: argparse.Namespace,
    class_names: List[str],
    device: torch.device
) -> Dict:
    """
    Train a single model and return best metrics.
    
    CRITICAL: Uses broadcast for early stopping to prevent NCCL timeout.
    """

    distributed = dist.is_initialized()
    rank = get_rank()
    num_classes = len(class_names)
    img_size = args.img_size

    print_rank0(f"\n{'='*60}")
    print_rank0(f"Training: {model_name.upper()}")
    print_rank0(f"Image size: {img_size}")
    print_rank0(f"{'='*60}")

    # Create datasets
    train_dataset = GC10Dataset(
        args.data_dir, 'train',
        get_transforms(img_size, is_train=True)
    )
    val_dataset = GC10Dataset(
        args.data_dir, 'val',
        get_transforms(img_size, is_train=False)
    )

    # Create samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )

    # Build model (pass img_size for ViT and YOLOv8)
    model = build_model(model_name, num_classes, img_size).to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Count parameters
    base_model = model.module if distributed else model
    params = count_parameters(base_model)
    print_rank0(f"Parameters: {params:.2f}M")

    # Loss, optimizer, scheduler
    criterion = get_criterion(model_name, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    # Output directory
    save_dir = Path(args.output_dir) / model_name
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    best_acc = 0.0
    best_metrics = {}
    patience_counter = 0

    # CRITICAL: Stop signal tensor for broadcasting early stopping
    # This prevents NCCL timeout when rank 0 breaks but others don't
    stop_signal = torch.zeros(1, device=device)

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        # Update learning rate
        lr = scheduler.step(epoch)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, model_name, args.warmup_epochs
        )

        # Evaluate on rank 0
        if is_main_process():
            val_metrics = evaluate(model, val_loader, criterion, device, model_name)

            print_rank0(
                f"E{epoch+1:03d}/{args.epochs} | "
                f"LR:{lr:.2e} | "
                f"Train:{train_acc:.2f}% | "
                f"Val:{val_metrics['accuracy']:.2f}% | "
                f"Bal:{val_metrics['balanced_accuracy']:.2f}% | "
                f"F1:{val_metrics['macro_f1']:.4f}"
            )

            # Check for improvement
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                best_metrics['epoch'] = epoch + 1
                patience_counter = 0

                # Save best model
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'metrics': best_metrics,
                    'class_names': class_names,
                    'model_name': model_name,
                    'num_classes': num_classes,
                }, save_dir / 'best_model.pth')

                print_rank0(f"  --> Best model saved!")
            else:
                patience_counter += 1

            # Set stop signal if early stopping triggered
            if patience_counter >= args.patience:
                stop_signal[0] = 1.0

        # CRITICAL: Broadcast stop signal to ALL ranks
        if distributed:
            dist.broadcast(stop_signal, src=0)

        # ALL ranks check and break together
        if stop_signal.item() >= 1.0:
            print_rank0(f"Early stopping at epoch {epoch+1}")
            break

    # Finalize metrics
    best_metrics['model_name'] = model_name
    best_metrics['params_M'] = params

    # Cleanup
    del model, optimizer, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()

    # Synchronize before returning
    if distributed:
        try:
            dist.barrier()
        except Exception:
            pass

    return best_metrics


# =============================================================================
# Results Output
# =============================================================================

def print_results_table(results: List[Dict], output_dir: str, class_names: List[str]):
    """Print and save benchmark results"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by accuracy (descending)
    results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)

    # Print table
    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS (sorted by accuracy)")
    print("=" * 110)
    print(f"{'Rank':<5} {'Model':<18} {'Params':<8} {'Acc%':<8} {'BalAcc%':<10} "
          f"{'F1':<8} {'Prec':<8} {'Recall':<8} {'Epoch':<6}")
    print("-" * 110)

    for i, r in enumerate(results, 1):
        print(
            f"{i:<5} "
            f"{r.get('model_name', 'N/A'):<18} "
            f"{r.get('params_M', 0):<8.2f} "
            f"{r.get('accuracy', 0):<8.2f} "
            f"{r.get('balanced_accuracy', 0):<10.2f} "
            f"{r.get('macro_f1', 0):<8.4f} "
            f"{r.get('macro_precision', 0):<8.4f} "
            f"{r.get('macro_recall', 0):<8.4f} "
            f"{r.get('epoch', 0):<6}"
        )

    print("=" * 110)

    # Save JSON
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save LaTeX table
    latex = generate_latex_table(results)
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - benchmark_results.json")
    print(f"  - results_table.tex")


def generate_latex_table(results: List[Dict]) -> str:
    """Generate LaTeX table for paper"""

    latex = r"""\begin{table}[H]
\centering
\caption{Performance comparison on GC10-DET dataset. Best results in \textbf{bold}.}
\label{tab:gc10_benchmark}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{Accuracy (\%)} & \textbf{Balanced Acc (\%)} & \textbf{Macro F1} \\
\midrule
"""

    # Find best values
    best_acc = max(r.get('accuracy', 0) for r in results) if results else 0
    best_bal = max(r.get('balanced_accuracy', 0) for r in results) if results else 0
    best_f1 = max(r.get('macro_f1', 0) for r in results) if results else 0

    for r in results:
        name = r.get('model_name', 'N/A').replace('_', r'\_')
        params = r.get('params_M', 0)
        acc = r.get('accuracy', 0)
        bal = r.get('balanced_accuracy', 0)
        f1 = r.get('macro_f1', 0)

        # Bold best values
        acc_str = f"\\textbf{{{acc:.2f}}}" if abs(acc - best_acc) < 0.01 else f"{acc:.2f}"
        bal_str = f"\\textbf{{{bal:.2f}}}" if abs(bal - best_bal) < 0.01 else f"{bal:.2f}"
        f1_str = f"\\textbf{{{f1:.4f}}}" if abs(f1 - best_f1) < 0.0001 else f"{f1:.4f}"

        latex += f"{name} & {params:.2f} & {acc_str} & {bal_str} & {f1_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(args: argparse.Namespace):
    """Run benchmark on all specified models"""

    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if distributed:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        print_rank0(f"Distributed training: {world_size} GPUs")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print_rank0(f"Single device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get class names from dataset
    temp_dataset = GC10Dataset(args.data_dir, split='train', transform=None)
    class_names = temp_dataset.classes
    num_classes = len(class_names)
    del temp_dataset

    print_rank0(f"\n{'='*60}")
    print_rank0("GC10-DET BENCHMARK")
    print_rank0(f"{'='*60}")
    print_rank0(f"Classes: {num_classes}")
    print_rank0(f"Class names: {class_names}")
    print_rank0(f"Image size: {args.img_size}")
    print_rank0(f"Batch size: {args.batch_size}")
    print_rank0(f"Epochs: {args.epochs}")
    print_rank0(f"Patience: {args.patience}")

    # Determine models to run
    if args.models:
        models_to_run = args.models
    else:
        # Default: paper comparison set
        models_to_run = DEFAULT_BENCHMARK_MODELS.copy()

    print_rank0(f"Models to benchmark: {models_to_run}")

    # Run benchmark
    results = []

    for model_name in models_to_run:
        # Synchronize before starting new model
        if distributed:
            try:
                dist.barrier()
            except Exception:
                pass

        try:
            # Check availability
            if model_name in ['saad_net', 'saad_net_no_cdl', 'saad', 'defect_lock', 'defect_lock_v2_imp']:
                if not SAAD_NET_AVAILABLE:
                    print_rank0(f"[SKIP] {model_name}: defect_lock_v2_improved.py not found")
                    continue

            if model_name.startswith('yolov8'):
                if not YOLO_AVAILABLE:
                    print_rank0(f"[SKIP] {model_name}: ultralytics not available")
                    continue

            if model_name in ['vit_b_16', 'vit_l_16'] and not TIMM_AVAILABLE:
                print_rank0(f"[WARNING] {model_name}: timm not available, using torchvision (224x224 only)")

            metrics = train_single_model(model_name, args, class_names, device)
            results.append(metrics)

            # Clear GPU memory
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print_rank0(f"[ERROR] {model_name}: {e}")
            import traceback
            if is_main_process():
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

            # Synchronize after error
            if distributed:
                try:
                    dist.barrier()
                except Exception:
                    pass

    # Print results
    if is_main_process() and results:
        print_results_table(results, args.output_dir, class_names)

    # Cleanup
    cleanup_distributed()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GC10-DET Benchmark - Multi-model comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(AVAILABLE_MODELS)}

Examples:
  # All default models (paper comparison)
  torchrun --nproc_per_node=4 gc10_benchmark.py --data_dir /path/to/gc10

  # Specific models only
  torchrun --nproc_per_node=4 gc10_benchmark.py \\
      --data_dir /path/to/gc10 \\
      --models swin_v2_t vit_b_16 saad_net saad_net_no_cdl

  # Single GPU test
  python gc10_benchmark.py --data_dir /path/to/gc10 --models efficientnet_b4
"""
    )

    # Required
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to GC10-DET dataset (with train/val folders)')

    # Optional
    parser.add_argument('--output_dir', type=str, default='./gc10_benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to benchmark. Default (paper comparison): '
                             'yolov8_cls_s, yolov8_cls_m, efficientnet_b4, swin_v2_t, '
                             'vit_b_16, saad_net_no_cdl, saad_net')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='Input image size (default: 1024 for Swin/ViT compatibility)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set by torchrun)')

    args = parser.parse_args()

    # Signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print_rank0("\n[INTERRUPTED] Cleaning up...")
        cleanup_distributed()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run benchmark
    try:
        run_benchmark(args)
    except KeyboardInterrupt:
        print_rank0("\n[INTERRUPTED] Cleaning up...")
    except Exception as e:
        print_rank0(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()