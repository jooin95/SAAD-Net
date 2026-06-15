#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNN/Transformer Baselines Training Script with Multi-GPU DDP Support
=====================================================================
Supports: ResNet-50, EfficientNet-B4, ViT-Base, Swin-T
For comparison experiments in DefectLoCK paper

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_baselines_ddp.py \
        --data_dir /path/to/data --save_dir ./experiments --models efficientnet_b4 vit_base swin_t
"""

import os
import sys

# NCCL configuration - must be before torch import
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
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
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import timm
except ImportError:
    print("Error: timm not installed. Run: pip install timm")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None


# ============================================================================
# Distributed Utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' not in os.environ:
        return 0, 1, False
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30)
    )
    
    return local_rank, world_size, True


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    """Get total number of processes"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


# ============================================================================
# Dataset
# ============================================================================

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        local_classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys())
        else:
            self.classes = local_classes
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in local_classes:
            if class_name not in self.class_to_idx:
                continue
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# Models
# ============================================================================

class ResNet50Classifier(nn.Module):
    """ResNet-50 with pretrained ImageNet weights"""
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class EfficientNetB4Classifier(nn.Module):
    """EfficientNet-B4 with pretrained ImageNet weights"""
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
    
    def forward(self, x):
        return self.backbone(x)


class ViTBaseClassifier(nn.Module):
    """ViT-Base/16 with pretrained ImageNet weights"""
    def __init__(self, num_classes, pretrained=True, dropout=0.3, img_size=1024):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            img_size=img_size
        )
    
    def forward(self, x):
        return self.backbone(x)


class SwinTClassifier(nn.Module):
    """Swin Transformer Tiny with pretrained ImageNet weights"""
    def __init__(self, num_classes, pretrained=True, dropout=0.3, img_size=1024):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            img_size=img_size
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# Training Functions
# ============================================================================

def get_transforms(img_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, loader, criterion, optimizer, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    if is_main_process():
        pbar = tqdm(loader, desc=f"Training")
    else:
        pbar = loader
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
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


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    
    all_preds = []
    all_labels = []
    
    if is_main_process():
        pbar = tqdm(loader, desc="Evaluating")
    else:
        pbar = loader
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            class_total[label] = class_total.get(label, 0) + 1
            if pred == label:
                class_correct[label] = class_correct.get(label, 0) + 1
    
    class_acc = {}
    for k in sorted(class_total.keys()):
        class_acc[class_names[k]] = 100. * class_correct.get(k, 0) / class_total[k]
    
    return total_loss / len(loader), 100. * correct / total, class_acc, all_preds, all_labels


def plot_results(history, save_dir, model_name):
    """Plot training results"""
    if plt is None:
        return
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    best_idx = np.argmax(history['val_acc'])
    axes[1].scatter([epochs[best_idx]], [history['val_acc'][best_idx]], 
                    color='gold', s=100, marker='*', zorder=5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{model_name} - Best: {history["val_acc"][best_idx]:.2f}%')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_results.png', dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, class_names, save_dir, model_name):
    """Plot confusion matrix"""
    if plt is None or confusion_matrix is None:
        return
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, data, title, fmt in [(axes[0], cm, 'Counts', 'd'), 
                                   (axes[1], cm_norm, 'Normalized', '.2f')]:
        im = ax.imshow(data, cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names,
               title=f'{model_name} - {title}', ylabel='True', xlabel='Predicted')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        thresh = data.max() / 2.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                       color="white" if data[i, j] > thresh else "black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name}_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_model(model_name, model, train_loader, val_loader, train_sampler,
                args, save_dir, class_names, device, distributed):
    """Train a single model"""
    
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # Cosine annealing with warmup
    warmup_epochs = args.warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()
    
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_preds, best_labels = [], []
    
    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"LR: {args.lr}, Batch: {args.batch_size} x {get_world_size()} GPUs")
        print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs} (lr: {current_lr:.2e})")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
        val_loss, val_acc, class_acc, preds, labels = evaluate(model, val_loader, criterion, device, class_names)
        
        scheduler.step()
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print("Class Accuracies:")
            for cls_name, acc in class_acc.items():
                print(f"  {cls_name}: {acc:.2f}%")
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_preds, best_labels = preds, labels
                
                # Save model (unwrap DDP if needed)
                model_to_save = model.module if distributed else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'val_acc': val_acc,
                    'class_acc': class_acc,
                    'class_names': class_names,
                    'model_name': model_name
                }, save_dir / f'{model_name}_best.pth')
                print(f"*** Best model saved! Acc: {val_acc:.2f}% ***")
            else:
                patience_counter += 1
        
        # Broadcast patience to all ranks
        if distributed:
            patience_tensor = torch.tensor([patience_counter], device=device)
            dist.broadcast(patience_tensor, src=0)
            patience_counter = int(patience_tensor.item())
        
        if patience_counter >= args.patience:
            if is_main_process():
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        if distributed:
            dist.barrier()
    
    # Save plots and summary (rank 0 only)
    if is_main_process():
        plot_results(history, save_dir, model_name)
        if best_preds and best_labels:
            plot_confusion(best_labels, best_preds, class_names, save_dir, model_name)
        
        summary = {
            'model': model_name,
            'best_accuracy': best_acc,
            'best_epoch': int(np.argmax(history['val_acc'])) + 1 if history['val_acc'] else 0,
            'epochs_trained': len(history['train_loss']),
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'world_size': get_world_size()
        }
        with open(save_dir / f'{model_name}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Baseline Training with DDP')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['efficientnet_b4'],
                        choices=['resnet50', 'efficientnet_b4', 'vit_base', 'swin_t'])
    args = parser.parse_args()
    
    # Setup distributed
    local_rank, world_size, distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if distributed else 'cuda')
    
    if is_main_process():
        print(f"Distributed: {distributed}, World Size: {world_size}")
    
    try:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Datasets
        train_transform = get_transforms(args.img_size, is_train=True)
        val_transform = get_transforms(args.img_size, is_train=False)
        
        train_path = os.path.join(args.data_dir, 'train')
        val_path = os.path.join(args.data_dir, 'val')
        if not os.path.exists(val_path):
            val_path = os.path.join(args.data_dir, 'test')
        
        train_dataset = ClassificationDataset(train_path, train_transform)
        val_dataset = ClassificationDataset(val_path, val_transform, class_to_idx=train_dataset.class_to_idx)
        
        class_names = train_dataset.classes
        num_classes = len(class_names)
        
        if is_main_process():
            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            print(f"Classes: {class_names}")
        
        # Samplers
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=4, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, sampler=val_sampler,
            num_workers=4, pin_memory=True
        )
        
        # Train each model
        results = {}
        
        for model_name in args.models:
            if is_main_process():
                print(f"\n{'='*70}")
                print(f"Initializing {model_name}...")
                print(f"{'='*70}")
            
            # Create model
            if model_name == 'resnet50':
                model = ResNet50Classifier(num_classes, pretrained=True)
            elif model_name == 'efficientnet_b4':
                model = EfficientNetB4Classifier(num_classes, pretrained=True)
            elif model_name == 'vit_base':
                model = ViTBaseClassifier(num_classes, pretrained=True, img_size=args.img_size)
            elif model_name == 'swin_t':
                model = SwinTClassifier(num_classes, pretrained=True, img_size=args.img_size)
            else:
                continue
            
            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            if is_main_process():
                print(f"{model_name}: {num_params:.1f}M parameters")
            
            # Move to device
            model = model.to(device)
            
            # Wrap with DDP
            if distributed:
                model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
            
            # Train
            best_acc = train_model(
                model_name, model, train_loader, val_loader, train_sampler,
                args, save_dir, class_names, device, distributed
            )
            results[model_name] = best_acc
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
            if distributed:
                dist.barrier()
        
        # Final results
        if is_main_process():
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            for name, acc in results.items():
                print(f"{name}: {acc:.2f}%")
            print("="*60)
            
            with open(save_dir / 'all_results.json', 'w') as f:
                json.dump(results, f, indent=2)
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()