#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-GPU Transformer Training Script
- Supports standard and enhanced Swin V2 models
- Distributed Data Parallel (DDP) training
- Mixed precision, Mixup/CutMix augmentation
"""

import os

# NCCL configuration (must be before torch import)
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# ============================================================================
# Imports
# ============================================================================
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
from typing import Dict, List, Tuple, Optional
import argparse
import math
import warnings
import signal
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# Import model modules
from transformer_classifiers import (
    get_transformer_classifier,
    FocalLoss as BaseFocalLoss
)

# Try to import enhanced models
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
    print("Warning: swin_v2_enhanced.py not found. Enhanced models disabled.")
    FocalLoss = BaseFocalLoss


# ============================================================================
# Distributed Training Utilities
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
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


# ============================================================================
# Dataset
# ============================================================================

class DefectDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Transforms
# ============================================================================

def get_transforms(img_size: int = 384, is_train: bool = True):
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
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ============================================================================
# Learning Rate Scheduler with Warmup
# ============================================================================

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Scheduler"""
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        warmup_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# ============================================================================
# Mixup & CutMix
# ============================================================================

def mixup_data(x, y, alpha=0.8):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# Plotting
# ============================================================================

def plot_results(history: Dict, save_dir: Path, class_names: List[str] = None):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Results', fontsize=16, fontweight='bold')

    # Train/Val Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Train/Val Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Best metrics text
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc'])
    best_loss = min(history['val_loss'])

    text_content = f"""
    Best Results
    -----------------
    Best Epoch: {best_epoch}
    Best Val Acc: {best_acc:.2f}%
    Best Val Loss: {best_loss:.4f}
    Final Train Acc: {history['train_acc'][-1]:.2f}%
    Final Val Acc: {history['val_acc'][-1]:.2f}%
    """

    axes[1, 1].text(0.1, 0.5, text_content, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary')

    plt.tight_layout()
    plt.savefig(save_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Individual plots
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Val Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Val Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    # LR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['lr'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'lr.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}")


# ============================================================================
# Training Function
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_mixup: bool = True,
    world_size: int = 1,
    use_arcface: bool = False
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if is_main_process():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training")
    else:
        pbar = loader

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup/CutMix (not used with ArcFace)
        use_aug = use_mixup and not use_arcface and np.random.rand() < 0.5
        if use_aug:
            if np.random.rand() < 0.5:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            else:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # ArcFace needs labels during forward
                if use_arcface:
                    outputs = model(images, labels)
                else:
                    outputs = model(images)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                if use_aug:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_arcface:
                outputs = model(images, labels)
            else:
                outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if use_aug:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

        if not use_aug:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if is_main_process():
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/max(total, 1):.2f}%'
            })

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / max(total, 1)

    if world_size > 1:
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        accuracy_tensor = torch.tensor([accuracy], device=device)
        avg_loss = reduce_tensor(avg_loss_tensor, world_size).item()
        accuracy = reduce_tensor(accuracy_tensor, world_size).item()

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    world_size: int = 1
) -> Tuple[float, float, Dict]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = {}
    class_total = {}

    if is_main_process():
        pbar = tqdm(loader, desc="Evaluating")
    else:
        pbar = loader

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Inference mode (no labels for ArcFace)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    if world_size > 1:
        total_loss_tensor = torch.tensor([total_loss], device=device)
        correct_tensor = torch.tensor([correct], device=device)
        total_tensor = torch.tensor([total], device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        total_loss = total_loss_tensor.item() / world_size
        correct = int(correct_tensor.item())
        total = int(total_tensor.item())

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    class_acc = {
        k: 100. * class_correct.get(k, 0) / class_total.get(k, 1)
        for k in class_total.keys()
    }

    return avg_loss, accuracy, class_acc


# ============================================================================
# Model Builder
# ============================================================================

def build_model(
    model_name: str,
    num_classes: int,
    model_kwargs: Dict = None
) -> nn.Module:
    """Build model based on model name"""
    model_kwargs = model_kwargs or {}

    # Enhanced models
    if model_name == 'swin_v2_enhanced':
        if not ENHANCED_AVAILABLE:
            raise ImportError("swin_v2_enhanced.py not found!")
        return SwinV2EnhancedClassifier(
            num_classes=num_classes,
            pretrained=True,
            **model_kwargs
        )

    elif model_name == 'swin_v2_arcface':
        if not ENHANCED_AVAILABLE:
            raise ImportError("swin_v2_enhanced.py not found!")
        return SwinV2ArcFaceClassifier(
            num_classes=num_classes,
            pretrained=True,
            **model_kwargs
        )

    # Standard models from transformer_classifiers.py
    else:
        return get_transformer_classifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            **model_kwargs
        )


# ============================================================================
# Main Training Loop
# ============================================================================

def train(
    data_dir: str,
    model_name: str = 'swin_v2',
    num_classes: int = 4,
    img_size: int = 384,
    batch_size: int = 16,
    num_epochs: int = 100,
    lr: float = 5e-5,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    use_mixup: bool = True,
    save_dir: str = './checkpoints',
    model_kwargs: Dict = None,
    resume: str = None,
    sync_bn: bool = True
):
    distributed = dist.is_initialized()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}' if distributed else 'cuda')

    # Check if using ArcFace model
    use_arcface = (model_name == 'swin_v2_arcface')

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Multi-GPU Training Configuration")
    print_rank0(f"{'='*70}")
    print_rank0(f"Model: {model_name}")
    print_rank0(f"Distributed: {distributed}")
    print_rank0(f"World size (GPU count): {world_size}")
    print_rank0(f"Current rank: {rank}")
    print_rank0(f"Device: {device}")
    print_rank0(f"Batch size per GPU: {batch_size}")
    print_rank0(f"Effective batch size: {batch_size * world_size}")
    print_rank0(f"Image size: {img_size}")
    print_rank0(f"Use ArcFace: {use_arcface}")
    print_rank0(f"{'='*70}\n")

    save_dir = Path(save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_transform = get_transforms(img_size, is_train=True)
    val_transform = get_transforms(img_size, is_train=False)

    # Datasets
    train_dataset = DefectDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    val_dataset = DefectDataset(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    class_names = train_dataset.classes

    # Distributed Sampler
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Build model
    model_kwargs = model_kwargs or {}
    model = build_model(model_name, num_classes, model_kwargs)

    # SyncBatchNorm
    if distributed and sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print_rank0("Using SyncBatchNorm")

    model = model.to(device)

    # DDP wrapper
    if distributed:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )

    # Parameter count
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Total parameters: {total_params:.2f}M")
        print(f"Trainable parameters: {trainable_params:.2f}M")

    # Loss function
    criterion = FocalLoss(gamma=2.0)

    # Optimizer with linear LR scaling
    scaled_lr = lr * world_size
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    print_rank0(f"Base LR: {lr}, Scaled LR: {scaled_lr}")

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=num_epochs,
        min_lr=1e-6
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0

    if resume and os.path.exists(resume):
        print_rank0(f"Resuming from {resume}")
        checkpoint = torch.load(resume, map_location=device)

        state_dict = checkpoint['model_state_dict']
        if distributed and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not distributed and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print_rank0(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}%")

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Starting training: {model_name}")
    print_rank0(f"{'='*70}\n")

    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        # Learning rate update
        current_lr = scheduler.step(epoch)

        print_rank0(f"\nEpoch {epoch+1}/{num_epochs} (lr: {current_lr:.2e})")
        print_rank0("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, scaler, use_mixup, world_size, use_arcface
        )

        # Validate
        val_loss, val_acc, class_acc = evaluate(
            model, val_loader, criterion, device, world_size
        )

        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("Class-wise Accuracy:")
            for idx, cls_name in enumerate(class_names):
                if idx in class_acc:
                    print(f"  {cls_name}: {class_acc[idx]:.2f}%")

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc

                model_state = model.module.state_dict() if distributed else model.state_dict()

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                    'class_names': class_names,
                    'model_name': model_name,
                    'img_size': img_size,
                    'model_kwargs': model_kwargs
                }
                torch.save(checkpoint, save_dir / 'best_model.pth')
                print(f"*** Best model saved! Acc: {val_acc:.2f}% ***")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                model_state = model.module.state_dict() if distributed else model.state_dict()
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                    'class_names': class_names,
                    'model_name': model_name,
                    'img_size': img_size,
                    'model_kwargs': model_kwargs
                }
                torch.save(checkpoint, save_dir / f'checkpoint_epoch{epoch+1}.pth')

        if distributed:
            dist.barrier()

    # Save final model and results
    if is_main_process():
        model_state = model.module.state_dict() if distributed else model.state_dict()
        checkpoint = {
            'epoch': num_epochs - 1,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'best_acc': best_acc,
            'class_names': class_names,
            'model_name': model_name,
            'img_size': img_size,
            'model_kwargs': model_kwargs
        }
        torch.save(checkpoint, save_dir / 'final_model.pth')

        # Save history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f)

        # Plot results
        plot_results(history, save_dir, class_names)

        # Save config
        config = {
            'model_name': model_name,
            'num_classes': num_classes,
            'img_size': img_size,
            'batch_size': batch_size,
            'effective_batch_size': batch_size * world_size,
            'num_epochs': num_epochs,
            'lr': lr,
            'scaled_lr': scaled_lr,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'best_acc': best_acc,
            'class_names': class_names,
            'world_size': world_size,
            'model_kwargs': model_kwargs
        }
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Training completed! Best accuracy: {best_acc:.2f}%")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*70}")

    return model, history


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Transformer Training')

    # Data
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')

    # Model
    parser.add_argument('--model', type=str, default='swin_v2',
                        choices=[
                            # Standard models
                            'swin_v2', 'maxvit', 'vit_attention_pool',
                            'pvtv2', 'efficient_vit', 'hybrid_cnn_transformer',
                            # Enhanced models
                            'swin_v2_enhanced', 'swin_v2_arcface'
                        ],
                        help='Model architecture')
    parser.add_argument('--model_size', type=str, default='t',
                        choices=['t', 's', 'b'],
                        help='Model size (t=tiny, s=small, b=base)')
    parser.add_argument('--num_classes', type=int, default=4)

    # Training
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--no_sync_bn', action='store_true')

    # Enhanced model options
    parser.add_argument('--use_cbam', action='store_true', default=True,
                        help='Use CBAM attention (for enhanced models)')
    parser.add_argument('--use_local_enhance', action='store_true', default=True,
                        help='Use local enhancement (for enhanced models)')
    parser.add_argument('--arcface_s', type=float, default=30.0,
                        help='ArcFace scale parameter')
    parser.add_argument('--arcface_m', type=float, default=0.5,
                        help='ArcFace margin parameter')

    # Checkpoint
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)

    # DDP
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        print(f"Initializing distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        setup_distributed(local_rank, world_size)
    else:
        print("Running in single GPU mode")

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print_rank0("\nTraining interrupted. Cleaning up...")
        cleanup_distributed()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Build model kwargs based on model type
        model_kwargs = {}

        if args.model == 'swin_v2':
            model_kwargs['model_size'] = args.model_size
            model_kwargs['use_multi_scale'] = False  # Disabled for stability

        elif args.model == 'swin_v2_enhanced':
            model_kwargs['model_size'] = args.model_size
            model_kwargs['use_cbam'] = args.use_cbam
            model_kwargs['use_local_enhance'] = args.use_local_enhance
            model_kwargs['use_fpn'] = False  # Simplified
            model_kwargs['use_deep_supervision'] = False

        elif args.model == 'swin_v2_arcface':
            model_kwargs['model_size'] = args.model_size
            model_kwargs['use_cbam'] = args.use_cbam
            model_kwargs['arcface_s'] = args.arcface_s
            model_kwargs['arcface_m'] = args.arcface_m

        elif args.model == 'vit_attention_pool':
            model_kwargs['model_size'] = args.model_size

        elif args.model in ['pvtv2', 'efficient_vit']:
            model_kwargs['img_size'] = args.img_size

        # Start training
        train(
            data_dir=args.data_dir,
            model_name=args.model,
            num_classes=args.num_classes,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup,
            use_mixup=not args.no_mixup,
            save_dir=args.save_dir,
            model_kwargs=model_kwargs,
            resume=args.resume,
            sync_bn=not args.no_sync_bn
        )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()