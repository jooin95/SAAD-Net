#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Training Script for Classification and Anomaly Detection
================================================================
Supports:
1. Classification Models (Swin V2, MaxViT, EfficientViT, etc.)
2. Anomaly Detection Models (EfficientAD, SimpleNet, PatchCore, STFPM)
3. Large Image Processing with Tile-based approach
4. Multi-GPU Distributed Training
"""

import os

# NCCL configuration
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_DEBUG'] = 'INFO'
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
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import argparse
import math
import warnings
import signal
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# Import models
from transformer_classifiers import get_transformer_classifier, FocalLoss as BaseFocalLoss

try:
    from swin_v2_enhanc import (
        SwinV2EnhancedClassifier,
        SwinV2ArcFaceClassifier,
        FocalLoss,
        CombinedLoss
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    FocalLoss = BaseFocalLoss

try:
    from anomaly_detection_models import (
        EfficientAD,
        SimpleNet,
        PatchCore,
        STFPM,
        get_anomaly_detector
    )
    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False
    print("Warning: anomaly_detection_models.py not found")

try:
    from highres_anomaly_models import (
        ReverseDistillation,
        FastFlow,
        DRAEM,
        get_highres_anomaly_detector
    )
    HIGHRES_ANOMALY_AVAILABLE = True
except ImportError:
    HIGHRES_ANOMALY_AVAILABLE = False
    print("Warning: highres_anomaly_models.py not found")

try:
    from highres_classification_models import (
        DINOv2Classifier,
        EVA02Classifier,
        InternImageClassifier,
        FocalNetClassifier,
        ConvNeXtV2Classifier,
        get_highres_classifier
    )
    HIGHRES_CLASS_AVAILABLE = True
except ImportError:
    HIGHRES_CLASS_AVAILABLE = False
    print("Warning: highres_classification_models.py not found")

try:
    from swin_v2_ultra_enhanced import (
        SwinV2UltraEnhanced,
        get_swin_v2_ultra,
        SAM, EMA,
        mixup_data, cutmix_data, mixup_criterion,
        FocalLoss as UltraFocalLoss
    )
    ULTRA_AVAILABLE = True
except ImportError as e:
    ULTRA_AVAILABLE = False
    print(f"Warning: swin_v2_ultra_enhanced.py import failed - {e}")

try:
    from swin_v2_small_defect import (
        SwinV2SmallDefectV2,
        get_swin_v2_small_defect,
        FocalLoss as SmallDefectFocalLoss
    )
    SMALL_DEFECT_AVAILABLE = True
except ImportError as e:
    SMALL_DEFECT_AVAILABLE = False
    print(f"Warning: swin_v2_small_defect.py import failed - {e}")

try:
    from swin_v2_overlock import (
        SwinV2OverLoCK,
        get_swin_v2_overlock,
        OverLoCKLoss
    )
    OVERLOCK_AVAILABLE = True
except ImportError as e:
    OVERLOCK_AVAILABLE = False
    print(f"Warning: swin_v2_overlock.py import failed - {e}")

try:
    from defect_lock import (
        DefectLoCK,
        get_defect_lock,
        DefectLoCKLoss
    )
    DEFECT_LOCK_AVAILABLE = True
except ImportError as e:
    DEFECT_LOCK_AVAILABLE = False
    print(f"Warning: defect_lock.py import failed - {e}")

try:
    from highres_classifiers import (
        DINOv2Classifier,
        EVA02Classifier,
        FocalNetClassifier,
        ConvNeXtV2Classifier,
        DETRStyleClassifier,
        get_highres_classifier
    )
    HIGHRES_CLASSIFIER_AVAILABLE = True
except ImportError:
    HIGHRES_CLASSIFIER_AVAILABLE = False
    print("Warning: highres_classifiers.py not found")

try:
    from large_image_pipeline import (
        TileProcessor,
        CoarseToFineDetector,
        UnifiedInspectionPipeline
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("Warning: large_image_pipeline.py not found")


# ============================================================================
# Distributed Utilities
# ============================================================================

def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30)
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ============================================================================
# Datasets
# ============================================================================

class ClassificationDataset(Dataset):
    """Dataset for supervised classification"""
    def __init__(
        self,
        root_dir: str,
        transform=None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        if is_main_process():
            print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
            for cls in self.classes:
                count = sum(1 for s in self.samples if s[1] == self.class_to_idx[cls])
                print(f"  {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection (normal samples only for training)"""
    def __init__(
        self,
        root_dir: str,
        transform=None,
        is_train: bool = True,
        normal_class: str = 'good',
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.normal_class = normal_class
        self.extensions = extensions
        
        self.samples = []
        self.labels = []  # 0 = normal, 1 = anomaly
        
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            is_normal = class_dir.name.lower() == normal_class.lower()
            
            # During training, only use normal samples
            if is_train and not is_normal:
                continue
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append(str(img_path))
                    self.labels.append(0 if is_normal else 1)
        
        if is_main_process():
            normal_count = sum(1 for l in self.labels if l == 0)
            anomaly_count = sum(1 for l in self.labels if l == 1)
            print(f"Loaded {len(self.samples)} images")
            print(f"  Normal: {normal_count}, Anomaly: {anomaly_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================================
# Transforms
# ============================================================================

def get_classification_transforms(img_size: int = 384, is_train: bool = True):
    """Transforms for classification"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_anomaly_transforms(img_size: int = 256, is_train: bool = True):
    """Transforms for anomaly detection (lighter augmentation)"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = 1e-7 + (self.base_lr - 1e-7) * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    def __init__(self, patience: int = 30, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.patience <= 0:
            return False
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ============================================================================
# Training Results Visualization (YOLO-style)
# ============================================================================

def plot_training_results(history: Dict, save_dir: Path, class_names: List[str] = None):
    """
    Plot and save training results like YOLO
    Saves: results.png, results.csv
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Results', fontsize=14, fontweight='bold')
    
    # 1. Train/Val Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Train/Val Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Learning Rate
    ax = axes[0, 2]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Train Loss (smoothed)
    ax = axes[1, 0]
    smooth_train = smooth_curve(history['train_loss'])
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.3, linewidth=1)
    ax.plot(epochs, smooth_train, 'b-', label='Smoothed', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train Loss (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    # 5. Val Accuracy (with best marker)
    ax = axes[1, 1]
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2)
    best_idx = np.argmax(history['val_acc'])
    best_acc = history['val_acc'][best_idx]
    ax.scatter([epochs[best_idx]], [best_acc], color='gold', s=200, marker='*', 
               zorder=5, label=f'Best: {best_acc:.2f}%')
    ax.axhline(y=best_acc, color='gold', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Val Accuracy (Best: {best_acc:.2f}% @ Epoch {epochs[best_idx]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Class-wise accuracy (if available)
    ax = axes[1, 2]
    if 'class_acc_history' in history and history['class_acc_history']:
        last_class_acc = history['class_acc_history'][-1]
        classes = list(last_class_acc.keys())
        accs = [last_class_acc[c] for c in classes]
        
        if class_names:
            labels = [class_names[c] if c < len(class_names) else f'Class {c}' for c in classes]
        else:
            labels = [f'Class {c}' for c in classes]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
        bars = ax.bar(labels, accs, color=colors)
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Class-wise Accuracy (Final)')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'Class accuracy\nnot available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Class-wise Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results.csv
    import csv
    csv_path = save_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        for i in range(len(epochs)):
            writer.writerow([
                epochs[i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.2f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.2f}",
                f"{history['lr'][i]:.2e}"
            ])
    
    print(f"Results saved to {save_dir / 'results.png'}")
    print(f"CSV saved to {csv_path}")


def plot_confusion_matrix(y_true: List, y_pred: List, class_names: List[str], save_dir: Path):
    """Plot and save confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix (Counts)',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Normalized
    ax = axes[1]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix (Normalized)',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, format(cm_normalized[i, j], '.2f'),
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir / 'confusion_matrix.png'}")


def smooth_curve(values: List[float], weight: float = 0.9) -> List[float]:
    """Exponential moving average smoothing"""
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# ============================================================================
# Model Builders
# ============================================================================

def build_classification_model(
    model_name: str,
    num_classes: int,
    model_kwargs: Dict = None
) -> nn.Module:
    """Build classification model"""
    model_kwargs = model_kwargs or {}
    
    # Swin V2 Ultra Enhanced (Best)
    if model_name == 'swin_v2_ultra':
        if not ULTRA_AVAILABLE:
            raise ImportError("swin_v2_ultra_enhanced.py not found")
        config = model_kwargs.pop('config', 'default')
        return get_swin_v2_ultra(num_classes=num_classes, config=config, **model_kwargs)
    
    # Swin V2 Small Defect Detector (NEW - optimized for small defects)
    if model_name == 'swin_v2_small_defect':
        if not SMALL_DEFECT_AVAILABLE:
            raise ImportError("swin_v2_small_defect.py not found")
        config = model_kwargs.pop('config', 'default')
        return get_swin_v2_small_defect(num_classes=num_classes, config=config, **model_kwargs)
    
    # Swin V2 OverLoCK (NEW - Overview-first-Look-Closely-next architecture)
    if model_name == 'swin_v2_overlock':
        if not OVERLOCK_AVAILABLE:
            raise ImportError("swin_v2_overlock.py not found")
        return get_swin_v2_overlock(num_classes=num_classes, **model_kwargs)
    
    # DefectLoCK (NEW - Paper-ready defect detection architecture)
    if model_name == 'defect_lock':
        if not DEFECT_LOCK_AVAILABLE:
            raise ImportError("defect_lock.py not found")
        return get_defect_lock(num_classes=num_classes, **model_kwargs)
    
    # High-resolution Classification models (512-1024 recommended)
    highres_models = ['dinov2', 'eva02', 'internimage', 'focalnet', 'convnext_v2']
    
    if model_name in highres_models:
        if not HIGHRES_CLASS_AVAILABLE:
            raise ImportError("highres_classification_models.py not found")
        return get_highres_classifier(model_name, num_classes, **model_kwargs)
    
    # Enhanced Swin V2 models
    if model_name == 'swin_v2_enhanced':
        if not ENHANCED_AVAILABLE:
            raise ImportError("swin_v2_enhanc.py not found")
        return SwinV2EnhancedClassifier(num_classes=num_classes, pretrained=True, **model_kwargs)
    
    elif model_name == 'swin_v2_arcface':
        if not ENHANCED_AVAILABLE:
            raise ImportError("swin_v2_enhanc.py not found")
        return SwinV2ArcFaceClassifier(num_classes=num_classes, pretrained=True, **model_kwargs)
    
    # Basic Transformer models
    else:
        return get_transformer_classifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            **model_kwargs
        )


def build_anomaly_model(
    model_name: str,
    model_kwargs: Dict = None
) -> nn.Module:
    """Build anomaly detection model"""
    model_kwargs = model_kwargs or {}
    
    # High-resolution models (512-1024 recommended)
    highres_models = ['reverse_distillation', 'rd4ad', 'fastflow', 'draem']
    
    if model_name in highres_models:
        if not HIGHRES_ANOMALY_AVAILABLE:
            raise ImportError("highres_anomaly_models.py not found")
        return get_highres_anomaly_detector(model_name, **model_kwargs)
    
    # Low-resolution models (256-512 recommended)
    if not ANOMALY_AVAILABLE:
        raise ImportError("anomaly_detection_models.py not found")
    
    return get_anomaly_detector(model_name, **model_kwargs)


# ============================================================================
# Training Functions
# ============================================================================

def train_classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_arcface: bool = False
) -> Tuple[float, float]:
    """Train one epoch for classification"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training") if is_main_process() else loader
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            if use_arcface:
                outputs = model(images, labels)
            else:
                outputs = model(images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def train_classification_epoch_ultra(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_arcface: bool = False,
    use_sam: bool = False,  # Disabled - not stable with AMP
    use_mixup: bool = False,
    use_cutmix: bool = False,
    mixup_alpha: float = 0.2,  # Conservative default
    ema = None
) -> Tuple[float, float]:
    """
    Train one epoch with enhanced features
    - MixUp / CutMix augmentation (conservative)
    - EMA update
    
    Note: SAM is disabled due to AMP compatibility issues
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Import from ultra module
    if ULTRA_AVAILABLE:
        from swin_v2_ultra_enhanced import mixup_data, cutmix_data, mixup_criterion
    
    pbar = tqdm(loader, desc="Training") if is_main_process() else loader
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # MixUp or CutMix (30% probability, conservative)
        use_mix = (use_mixup or use_cutmix) and ULTRA_AVAILABLE and np.random.rand() < 0.3
        if use_mix:
            if use_cutmix and (not use_mixup or np.random.rand() < 0.5):
                images, labels_a, labels_b, lam = cutmix_data(images, labels, mixup_alpha)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        
        optimizer.zero_grad()
        
        # Standard training (SAM disabled for stability)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            
            # Handle different output types
            if isinstance(outputs, dict):
                # DefectLoCK returns dict with 'logits', 'aux_logits', 'embeddings', etc.
                logits = outputs['logits']
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    # DefectLoCKLoss expects dict outputs
                    if hasattr(criterion, 'forward') and 'DefectLoCK' in criterion.__class__.__name__:
                        loss_dict = criterion(outputs, labels)
                        loss = loss_dict['total']
                    else:
                        loss = criterion(logits, labels)
                outputs = logits  # For accuracy calculation
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
            else:
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # EMA update
        if ema is not None:
            ema.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        
        # For MixUp/CutMix, count correct for primary labels
        total += labels.size(0)
        correct += predicted.eq(labels_a).sum().item()
        
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def train_anomaly_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    model_name: str
) -> float:
    """Train one epoch for anomaly detection"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(loader, desc="Training") if is_main_process() else loader
    
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            if model_name == 'efficientad':
                losses = model.compute_loss(images)
                loss = losses['total']
            elif model_name in ['simplenet', 'stfpm']:
                loss = model.compute_loss(images)
            # High-resolution models
            elif model_name in ['reverse_distillation', 'rd4ad', 'fastflow']:
                loss = model.compute_loss(images)
            elif model_name == 'draem':
                # DRAEM requires synthetic anomaly images
                # Simple impl: original=normal, noise added=anomaly
                noise = torch.randn_like(images) * 0.1
                mask = (torch.rand(images.size(0), 1, images.size(2), images.size(3), device=device) > 0.7).float()
                anomaly_images = images + noise * mask
                loss = model.compute_loss(images, anomaly_images, mask)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict]:
    """Evaluate classification model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    pbar = tqdm(loader, desc="Evaluating") if is_main_process() else loader
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        
        # Handle different output types
        if isinstance(outputs, dict):
            logits = outputs['logits']
            # DefectLoCKLoss expects dict outputs
            if hasattr(criterion, 'forward') and 'DefectLoCK' in criterion.__class__.__name__:
                loss_dict = criterion(outputs, labels)
                loss = loss_dict['total']
            else:
                loss = criterion(logits, labels)
            outputs = logits
        elif isinstance(outputs, tuple):
            outputs = outputs[0]
            loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            class_total[label] = class_total.get(label, 0) + 1
            if pred == label:
                class_correct[label] = class_correct.get(label, 0) + 1
    
    class_acc = {k: 100. * class_correct.get(k, 0) / class_total[k] for k in class_total}
    
    return total_loss / len(loader), 100. * correct / total, class_acc


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[List, List]:
    """Get all predictions and labels for confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        
        outputs = model(images)
        
        # Handle different output types
        if isinstance(outputs, dict):
            outputs = outputs['logits']
        elif isinstance(outputs, tuple):
            outputs = outputs[0]
        
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    
    return all_preds, all_labels


@torch.no_grad()
def evaluate_anomaly(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict:
    """Evaluate anomaly detection model"""
    model.eval()
    
    scores = []
    labels = []
    
    pbar = tqdm(loader, desc="Evaluating") if is_main_process() else loader
    
    for images, batch_labels in pbar:
        images = images.to(device, non_blocking=True)
        
        output = model(images)
        if isinstance(output, tuple):
            score = output[0]
        else:
            score = output
        
        scores.extend(score.cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().tolist())
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, f1_score
    
    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.5
    
    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(scores.min(), scores.max(), 100):
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return {
        'auroc': auroc,
        'best_f1': best_f1,
        'best_threshold': best_thresh
    }


# ============================================================================
# Main Training Loops
# ============================================================================

def train_classification(args):
    """Main training loop for classification"""
    distributed = dist.is_initialized()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}' if distributed else 'cuda')
    
    use_arcface = args.model in ['swin_v2_arcface'] or (args.model == 'swin_v2_ultra' and args.ultra_config == 'arcface')
    use_ultra = args.model == 'swin_v2_ultra'
    
    print_rank0(f"\n{'='*70}")
    print_rank0(f"Classification Training: {args.model}")
    if use_ultra:
        print_rank0(f"Ultra Config: {args.ultra_config}")
        print_rank0(f"SAM: {args.use_sam}, EMA: {args.use_ema}, MixUp: {args.use_mixup}, CutMix: {args.use_cutmix}")
    print_rank0(f"{'='*70}")
    
    # Dataset
    train_transform = get_classification_transforms(args.img_size, is_train=True)
    val_transform = get_classification_transforms(args.img_size, is_train=False)
    
    train_dataset = ClassificationDataset(os.path.join(args.data_dir, 'train'), train_transform)
    val_dataset = ClassificationDataset(os.path.join(args.data_dir, 'val'), val_transform)
    
    class_names = train_dataset.classes
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True
    )
    
    # Model kwargs
    model_kwargs = {'model_size': args.model_size}
    if args.model == 'swin_v2_enhanced':
        model_kwargs.update({'use_cbam': True, 'use_local_enhance': True})
    elif args.model == 'swin_v2_arcface':
        model_kwargs.update({'arcface_s': 30.0, 'arcface_m': 0.5})
    # Swin V2 Ultra Enhanced
    elif args.model == 'swin_v2_ultra':
        model_kwargs = {
            'model_size': args.model_size,
            'config': args.ultra_config,
            'dropout': 0.3,
        }
    # Swin V2 Small Defect Detector (NEW)
    elif args.model == 'swin_v2_small_defect':
        model_kwargs = {
            'model_size': args.model_size,
            'config': args.defect_config,
            'dropout': 0.3,
        }
    # Swin V2 OverLoCK (NEW - OverLoCK-style architecture)
    elif args.model == 'swin_v2_overlock':
        model_kwargs = {
            'model_size': args.model_size,
            'drop_path_rate': 0.15 if args.model_size == 't' else 0.4,
            'use_aux_loss': True,
        }
    # DefectLoCK (Paper-ready architecture)
    elif args.model == 'defect_lock':
        model_kwargs = {
            'model_size': args.model_size,
            'use_contrastive': args.use_contrastive,
            'dropout': 0.3,
        }
    # High-resolution Classification models
    elif args.model == 'dinov2':
        model_kwargs = {'model_size': 'base', 'pretrained': True}
    elif args.model == 'eva02':
        model_kwargs = {'model_size': 'base', 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'internimage':
        model_kwargs = {'model_size': 'small', 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'focalnet':
        model_kwargs = {'model_size': 'base', 'pretrained': True}
    elif args.model == 'convnext_v2':
        model_kwargs = {'model_size': 'base', 'pretrained': True}
    
    model = build_classification_model(args.model, args.num_classes, model_kwargs)
    
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Loss function
    if args.model == 'defect_lock' and DEFECT_LOCK_AVAILABLE:
        criterion = DefectLoCKLoss(
            num_classes=args.num_classes,
            aux_weight=args.defect_lock_aux_weight,
            contrastive_weight=args.defect_lock_contrastive_weight,
            focal_gamma=2.0,
            label_smoothing=0.1
        )
        print_rank0(f"Using DefectLoCK Loss (aux={args.defect_lock_aux_weight}, contrastive={args.defect_lock_contrastive_weight})")
    elif args.model == 'swin_v2_overlock' and OVERLOCK_AVAILABLE:
        criterion = OverLoCKLoss(
            num_classes=args.num_classes,
            aux_weight=args.overlock_aux_weight,
            label_smoothing=0.1
        )
        print_rank0(f"Using OverLoCK Loss (aux_weight={args.overlock_aux_weight})")
    else:
        criterion = FocalLoss(gamma=2.0)
    
    # Optimizer setup
    if args.model in ['dinov2', 'eva02']:
        # DINOv2/EVA02: Backbone frozen, train head only
        if hasattr(model, 'module'):
            backbone = model.module.backbone
        else:
            backbone = model.backbone
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print_rank0(f"DINOv2/EVA02: Backbone frozen, training head only")
        print_rank0(f"Trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
        
        base_optimizer = torch.optim.AdamW
        optimizer_kwargs = {'lr': args.lr * 0.1 * world_size, 'weight_decay': 0.01}
        
        if args.use_sam and ULTRA_AVAILABLE:
            optimizer = SAM(trainable_params, base_optimizer, **optimizer_kwargs)
        else:
            optimizer = base_optimizer(trainable_params, **optimizer_kwargs)
    else:
        base_optimizer = torch.optim.AdamW
        optimizer_kwargs = {'lr': args.lr * world_size, 'weight_decay': 0.05}
        optimizer = base_optimizer(model.parameters(), **optimizer_kwargs)
    
    # EMA setup
    ema = None
    if args.use_ema and ULTRA_AVAILABLE:
        raw_model = model.module if distributed else model
        ema = EMA(raw_model, decay=0.9999)
        print_rank0("Using EMA (decay=0.9999)")
    
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='max')
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_acc = 0
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [], 
        'lr': [],
        'class_acc_history': []
    }
    
    all_preds = []
    all_labels = []
    
    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        lr = scheduler.step(epoch)
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs} (lr: {lr:.2e})")
        
        # Train with MixUp/CutMix
        train_loss, train_acc = train_classification_epoch_ultra(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_arcface=use_arcface,
            use_sam=args.use_sam and ULTRA_AVAILABLE,
            use_mixup=args.use_mixup and ULTRA_AVAILABLE,
            use_cutmix=args.use_cutmix and ULTRA_AVAILABLE,
            mixup_alpha=args.mixup_alpha,
            ema=ema
        )
        
        # Evaluate with EMA model if available
        if ema is not None:
            ema.apply_shadow()
        
        val_loss, val_acc, class_acc = evaluate_classification(model, val_loader, criterion, device)
        
        # Get predictions for confusion matrix (last epoch or best)
        preds, labels = get_predictions(model, val_loader, device)
        
        if ema is not None:
            ema.restore()
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Class-wise accuracy
            print("Class Accuracies:")
            for cls_idx, acc in class_acc.items():
                cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                print(f"  {cls_name}: {acc:.2f}%")
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(lr)
            history['class_acc_history'].append(class_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                all_preds = preds
                all_labels = labels
                
                # Save EMA weights if available
                if ema is not None:
                    ema.apply_shadow()
                
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'val_acc': val_acc,
                    'class_names': class_names,
                    'model_name': args.model,
                    'img_size': args.img_size,
                    'ultra_config': args.ultra_config if use_ultra else None
                }, save_dir / 'best_model.pth')
                
                if ema is not None:
                    ema.restore()
                
                print(f"*** Best model saved! Acc: {val_acc:.2f}% ***")
            
            # Save last model
            model_state = model.module.state_dict() if distributed else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'val_acc': val_acc,
                'class_names': class_names,
                'model_name': args.model,
                'img_size': args.img_size
            }, save_dir / 'last_model.pth')
            
            # Check early stopping
            if early_stopping(val_acc, epoch):
                print(f"\n*** Early stopping triggered at epoch {epoch+1} ***")
                print(f"Best accuracy: {best_acc:.2f}% at epoch {early_stopping.best_epoch + 1}")
                break
            
            # Print early stopping status
            if args.patience > 0:
                print(f"Early stopping: {early_stopping.counter}/{args.patience}")
    
    if is_main_process():
        # Save history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot and save results (YOLO-style)
        plot_training_results(history, save_dir, class_names)
        
        # Plot confusion matrix
        if all_preds and all_labels:
            plot_confusion_matrix(all_labels, all_preds, class_names, save_dir)
        
        # Save training summary
        summary = {
            'model': args.model,
            'model_size': args.model_size,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs_trained': len(history['train_loss']),
            'best_accuracy': best_acc,
            'best_epoch': int(np.argmax(history['val_acc'])) + 1,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'use_sam': args.use_sam,
            'use_ema': args.use_ema,
            'use_cutmix': args.use_cutmix,
            'use_mixup': args.use_mixup,
            'early_stopped': early_stopping.early_stop,
            'class_names': class_names
        }
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best accuracy: {best_acc:.2f}%")
        print(f"Results saved to: {save_dir}")
        print(f"  - best_model.pth")
        print(f"  - last_model.pth")
        print(f"  - results.png")
        print(f"  - results.csv")
        print(f"  - confusion_matrix.png")
        print(f"  - history.json")
        print(f"  - summary.json")
        print(f"{'='*60}")
    
    # Cleanup DDP
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def train_anomaly(args):
    """Main training loop for anomaly detection"""
    distributed = dist.is_initialized()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}' if distributed else 'cuda')
    
    print_rank0(f"\n{'='*70}")
    print_rank0(f"Anomaly Detection Training: {args.model}")
    print_rank0(f"{'='*70}")
    
    # Dataset
    train_transform = get_anomaly_transforms(args.img_size, is_train=True)
    val_transform = get_anomaly_transforms(args.img_size, is_train=False)
    
    train_dataset = AnomalyDataset(
        os.path.join(args.data_dir, 'train'), train_transform, 
        is_train=True, normal_class=args.normal_class
    )
    val_dataset = AnomalyDataset(
        os.path.join(args.data_dir, 'val'), val_transform,
        is_train=False, normal_class=args.normal_class
    )
    
    # Samplers
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=4, pin_memory=True
    )
    
    # Model
    model_kwargs = {}
    if args.model == 'efficientad':
        model_kwargs = {'model_size': 'small', 'use_autoencoder': True}
    elif args.model == 'simplenet':
        model_kwargs = {'backbone': 'wide_resnet50_2'}
    elif args.model == 'stfpm':
        model_kwargs = {'backbone': 'resnet18'}
    elif args.model == 'patchcore':
        model_kwargs = {'backbone': 'wide_resnet50_2', 'coreset_ratio': 0.01}
    # High-resolution models (512-1024 support)
    elif args.model in ['reverse_distillation', 'rd4ad']:
        model_kwargs = {'backbone': 'wide_resnet50_2', 'layers': ['layer1', 'layer2', 'layer3']}
    elif args.model == 'fastflow':
        model_kwargs = {'backbone': 'wide_resnet50_2', 'layers': ['layer2', 'layer3'], 'flow_steps': 8}
    elif args.model == 'draem':
        model_kwargs = {'base_channels': 64}
    
    model = build_anomaly_model(args.model, model_kwargs)
    model = model.to(device)
    
    # Special handling for PatchCore (no training, just feature extraction)
    if args.model == 'patchcore':
        print_rank0("PatchCore: Building memory bank...")
        model.fit(train_loader, device)
        
        print_rank0("Evaluating...")
        metrics = evaluate_anomaly(model, val_loader, device)
        print_rank0(f"AUROC: {metrics['auroc']:.4f}, Best F1: {metrics['best_f1']:.4f}")
        
        if is_main_process():
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'memory_bank': model.memory_bank,
                'model_name': args.model,
                'metrics': metrics
            }, save_dir / 'patchcore_model.pth')
        return
    
    # DDP for trainable models
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Set normalization params (only for certain models)
    if args.model in ['efficientad', 'simplenet']:
        print_rank0("Computing normalization parameters...")
        raw_model = model.module if distributed else model
        raw_model.set_normalization_params(train_loader, device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * world_size, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_auroc = 0
    history = {'train_loss': [], 'auroc': [], 'f1': [], 'lr': []}
    
    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        
        lr = scheduler.step(epoch)
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs} (lr: {lr:.2e})")
        
        train_loss = train_anomaly_epoch(model, train_loader, optimizer, device, scaler, args.model)
        
        metrics = evaluate_anomaly(model, val_loader, device)
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}")
            print(f"AUROC: {metrics['auroc']:.4f}, F1: {metrics['best_f1']:.4f}")
            
            history['train_loss'].append(train_loss)
            history['auroc'].append(metrics['auroc'])
            history['f1'].append(metrics['best_f1'])
            history['lr'].append(lr)
            
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'auroc': metrics['auroc'],
                    'threshold': metrics['best_threshold'],
                    'model_name': args.model
                }, save_dir / 'best_model.pth')
                print(f"*** Best model saved! AUROC: {metrics['auroc']:.4f} ***")
    
    if is_main_process():
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f)
        print(f"\nTraining completed! Best AUROC: {best_auroc:.4f}")
    
    # Cleanup DDP
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Training Script')
    
    # Task type
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'anomaly'],
                        help='Task type')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--normal_class', type=str, default='good',
                        help='Normal class name for anomaly detection')
    
    # Model
    parser.add_argument('--model', type=str, default='swin_v2',
                        choices=[
                            # Classification (Basic)
                            'swin_v2', 'swin_v2_enhanced', 'swin_v2_arcface',
                            'swin_v2_ultra',  # Ultra Enhanced (SAM, EMA, CutMix support)
                            'swin_v2_small_defect',  # Optimized for small defect detection
                            'swin_v2_overlock',  # OverLoCK-style architecture
                            'defect_lock',  # NEW: Paper-ready DefectLoCK architecture
                            'maxvit', 'vit_attention_pool', 'efficient_vit',
                            # Classification (High-resolution Transformer)
                            'dinov2', 'eva02', 'internimage', 'focalnet', 'convnext_v2',
                            # Anomaly Detection (Low-resolution 256-512)
                            'efficientad', 'simplenet', 'patchcore', 'stfpm',
                            # Anomaly Detection (High-resolution 512-1024)
                            'reverse_distillation', 'rd4ad', 'fastflow', 'draem'
                        ])
    parser.add_argument('--model_size', type=str, default='t', choices=['xt', 't', 's', 'b'])
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--ultra_config', type=str, default='default',
                        choices=['default', 'fpn', 'deep', 'arcface', 'full'],
                        help='Configuration for swin_v2_ultra model')
    parser.add_argument('--defect_config', type=str, default='default',
                        choices=['default', 'light', 'full'],
                        help='Configuration for swin_v2_small_defect model')
    parser.add_argument('--overlock_aux_weight', type=float, default=0.4,
                        help='Auxiliary loss weight for swin_v2_overlock model')
    parser.add_argument('--defect_lock_aux_weight', type=float, default=0.4,
                        help='Auxiliary loss weight for defect_lock model')
    parser.add_argument('--defect_lock_contrastive_weight', type=float, default=0.1,
                        help='Contrastive loss weight for defect_lock model')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                        help='Use contrastive learning for defect_lock')
    
    # Training
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup', type=int, default=5)
    
    # Ultra Enhanced Training Options
    parser.add_argument('--use_sam', action='store_true', help='Use SAM optimizer')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA')
    parser.add_argument('--use_mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--use_cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=1.0, help='MixUp/CutMix alpha')
    parser.add_argument('--use_tta', action='store_true', help='Use TTA for evaluation')
    
    # Early Stopping
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (0 to disable)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    
    # DDP
    parser.add_argument('--local_rank', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Setup distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        setup_distributed(local_rank, world_size)
    
    # Signal handler
    def signal_handler(sig, frame):
        print_rank0("\nInterrupted. Cleaning up...")
        cleanup_distributed()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.task == 'classification':
            train_classification(args)
        else:
            train_anomaly(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()