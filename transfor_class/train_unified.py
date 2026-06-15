#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

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
from typing import Dict, List, Tuple, Optional, Union
import argparse
import math
import warnings
import signal
import sys
import gc
import atexit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# timm for flexible ViT image sizes
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. ViT will use torchvision (224x224 only)")

from transformer_classifiers import get_transformer_classifier, FocalLoss as BaseFocalLoss

try:
    from swin_v2_enhanc import (SwinV2EnhancedClassifier, SwinV2ArcFaceClassifier, FocalLoss, CombinedLoss)
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    FocalLoss = BaseFocalLoss

try:
    from anomaly_detection_models import (EfficientAD, SimpleNet, PatchCore, STFPM, get_anomaly_detector)
    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False
    print("Warning: anomaly_detection_models.py not found")

try:
    from highres_anomaly_models import (ReverseDistillation, FastFlow, DRAEM, get_highres_anomaly_detector)
    HIGHRES_ANOMALY_AVAILABLE = True
except ImportError:
    HIGHRES_ANOMALY_AVAILABLE = False
    print("Warning: highres_anomaly_models.py not found")

try:
    from highres_classification_models import (DINOv2Classifier, EVA02Classifier, InternImageClassifier,
                                               FocalNetClassifier, ConvNeXtV2Classifier, get_highres_classifier)
    HIGHRES_CLASS_AVAILABLE = True
except ImportError:
    HIGHRES_CLASS_AVAILABLE = False
    print("Warning: highres_classification_models.py not found")

try:
    from swin_v2_ultra_enhanced import (SwinV2UltraEnhanced, get_swin_v2_ultra, SAM, EMA, TTAWrapper,
                                        mixup_data, cutmix_data, mixup_criterion,
                                        CombinedLoss as UltraCombinedLoss, FocalLoss as UltraFocalLoss)
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False
    print("Warning: swin_v2_ultra_enhanced.py not found")

try:
    from defect_lock_v2_improved import (DefectLoCKv2Improved, get_defect_lock_v2_improved,
                                          DefectLoCKv2ImprovedLoss, get_defect_lock_v2_improved_for_training)
    DEFECT_LOCK_V2_IMP_AVAILABLE = True
except ImportError as e:
    DEFECT_LOCK_V2_IMP_AVAILABLE = False
    print(f"Warning: defect_lock_v2_improved.py import failed - {e}")

try:
    from highres_classifiers import (DINOv2Classifier, EVA02Classifier, FocalNetClassifier,
                                     ConvNeXtV2Classifier, DETRStyleClassifier, get_highres_classifier)
    HIGHRES_CLASSIFIER_AVAILABLE = True
except ImportError:
    HIGHRES_CLASSIFIER_AVAILABLE = False
    print("Warning: highres_classifiers.py not found")

try:
    from saad_net import (SAADNet, get_saad_net, SAADNetLoss, get_saad_net_loss)
    SAAD_NET_AVAILABLE = True
except ImportError as e:
    SAAD_NET_AVAILABLE = False
    print(f"Warning: saad_net.py import failed - {e}")

try:
    from saad_net_v3 import (SAADNetV2, get_saad_net_v2, SAADNetV2Loss, get_saad_net_v2_loss)
    SAAD_NET_V2_AVAILABLE = True
except ImportError as e:
    SAAD_NET_V2_AVAILABLE = False
    print(f"Warning: saad_net_v3.py import failed - {e}")

try:
    from large_image_pipeline import (TileProcessor, CoarseToFineDetector, UnifiedInspectionPipeline)
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    print("Warning: large_image_pipeline.py not found")


# ============================================================================
# Distributed Utilities
# ============================================================================

_CLEANUP_DONE = False

def setup_distributed(rank, world_size, backend='nccl'):
    dist.init_process_group(backend=backend, init_method='env://',
                            world_size=world_size, rank=rank, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training - safe to call multiple times"""
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

atexit.register(cleanup_distributed)

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
    def __init__(self, root_dir, transform=None,
                 extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class AnomalyDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, normal_class='good',
                 extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir(): continue
            is_normal = class_dir.name.lower() == normal_class.lower()
            if is_train and not is_normal: continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    self.samples.append(str(img_path))
                    self.labels.append(0 if is_normal else 1)
        if is_main_process():
            print(f"Loaded {len(self.samples)} images  "
                  f"Normal:{sum(1 for l in self.labels if l==0)}  "
                  f"Anomaly:{sum(1 for l in self.labels if l==1)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]


class DetectionDataset(Dataset):
    GC10_CLASSES = [
        'punching_hole', 'welding_line', 'crescent_gap', 'water_spot',
        'oil_spot', 'silk_spot', 'inclusion', 'rolled_pit', 'crease',
        'waist_folding', 'good',
    ]

    def __init__(self, voc_split_dir, transform=None, use_hflip=False,
                 extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        import xml.etree.ElementTree as ET
        self._ET = ET
        self.transform = transform
        self.use_hflip = use_hflip
        self.extensions = extensions
        split_dir = Path(voc_split_dir)
        self.img_dir = split_dir / 'images'
        self.ann_dir = split_dir / 'annotations'
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.img_dir}")
        self.class_to_idx = {c: i for i, c in enumerate(self.GC10_CLASSES)}
        self.classes = self.GC10_CLASSES
        self.xml_index: Dict[str, str] = {}
        if self.ann_dir.exists():
            for f in self.ann_dir.rglob('*.xml'):
                self.xml_index[f.stem] = str(f)
        self.samples: List[str] = sorted([
            str(p) for p in self.img_dir.rglob('*') if p.suffix.lower() in extensions])
        ann_count = sum(1 for p in self.samples if Path(p).stem in self.xml_index)
        if is_main_process():
            print(f"[DetectionDataset] {split_dir.name}: {len(self.samples)} images  "
                  f"(annotated={ann_count}, good/no-ann={len(self.samples)-ann_count})")

    def __len__(self): return len(self.samples)

    def _parse_xml(self, xml_path, img_w, img_h):
        try:
            root = self._ET.parse(xml_path).getroot()
        except Exception:
            return torch.zeros((0, 5), dtype=torch.float32), self.class_to_idx['good']
        rows = []
        area_cls = {}
        for obj in root.findall('object'):
            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            if difficult: continue
            # CRITICAL: use 'is not None' instead of truthiness check.
            # bool(XML Element) = False when element has no children,
            # so 'obj.find("name") or obj.find("n")' incorrectly returns None
            # for valid <name>crease</name> elements -> all GT boxes are skipped!
            n_tag = obj.find('name')
            if n_tag is None:
                n_tag = obj.find('n')
            if n_tag is None:
                continue
            cls_name = n_tag.text.strip() if n_tag.text else ''
            cls_idx = self.class_to_idx.get(cls_name, -1)
            if cls_idx < 0: continue
            bb = obj.find('bndbox')
            xmin = max(0, int(float(bb.find('xmin').text)))
            ymin = max(0, int(float(bb.find('ymin').text)))
            xmax = min(img_w, int(float(bb.find('xmax').text)))
            ymax = min(img_h, int(float(bb.find('ymax').text)))
            if xmax <= xmin or ymax <= ymin: continue
            cx = (xmin + xmax) / 2.0 / img_w
            cy = (ymin + ymax) / 2.0 / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            rows.append([float(cls_idx), cx, cy, bw, bh])
            area_cls[cls_name] = area_cls.get(cls_name, 0.0) + (xmax-xmin)*(ymax-ymin)
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32), self.class_to_idx['good']
        img_cls_name = max(area_cls, key=area_cls.get)
        img_cls_idx  = self.class_to_idx.get(img_cls_name, self.class_to_idx['good'])
        return torch.tensor(rows, dtype=torch.float32), img_cls_idx

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        stem = Path(img_path).stem
        xml = self.xml_index.get(stem)
        if xml:
            boxes, cls_label = self._parse_xml(xml, w, h)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
            cls_label = self.class_to_idx['good']
        if self.use_hflip and torch.rand(1).item() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.shape[0] > 0:
                boxes = boxes.clone()
                boxes[:, 1] = 1.0 - boxes[:, 1]
        if self.transform:
            image = self.transform(image)
        return image, cls_label, boxes, img_path


def detection_collate_fn(batch):
    images     = torch.stack([b[0] for b in batch])
    cls_labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    boxes      = [b[2] for b in batch]
    img_paths  = [b[3] for b in batch]
    return images, cls_labels, boxes, img_paths


# ============================================================================
# Transforms
# ============================================================================

def get_classification_transforms(img_size=384, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
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


def get_anomaly_transforms(img_size=256, is_train=True):
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


def get_detection_transforms(img_size=512, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ============================================================================
# Schedulers & Early Stopping
# ============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = 1e-7 + (self.base_lr - 1e-7) * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.patience <= 0:
            return False
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        improved = (score > self.best_score + self.min_delta if self.mode == 'max'
                    else score < self.best_score - self.min_delta)
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
# Visualization
# ============================================================================

def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_results(history, save_dir, class_names=None):
    epochs   = list(range(1, len(history['train_loss']) + 1))
    has_map  = 'map50' in history and len(history['map50']) > 0
    ncols    = 4 if has_map else 3
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))
    fig.suptitle('Training Results', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'],   'r-', label='Val Loss',   linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, history['val_acc'],   'r-', label='Val Acc',   linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_title('LR Schedule')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    if has_map:
        ax = axes[0, 3]
        map_ep = list(range(1, len(history['map50']) + 1))
        vals   = [v * 100 for v in history['map50']]
        ax.plot(map_ep, vals, color='purple', linewidth=2)
        best_i = int(np.argmax(history['map50']))
        ax.scatter([map_ep[best_i]], [vals[best_i]], color='gold', s=200, marker='*', zorder=5,
                   label=f"Best: {vals[best_i]:.2f}%")
        ax.set_xlabel('Epoch'); ax.set_ylabel('mAP50 (%)')
        ax.set_title(f"mAP50 (Best: {vals[best_i]:.2f}% @ Ep{map_ep[best_i]})")
        ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    smooth_t = smooth_curve(history['train_loss'])
    ax.plot(epochs, history['train_loss'], 'b-', alpha=0.3, linewidth=1)
    ax.plot(epochs, smooth_t, 'b-', label='Smoothed', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Train Loss (Smoothed)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2)
    best_i   = int(np.argmax(history['val_acc']))
    best_acc = history['val_acc'][best_i]
    ax.scatter([epochs[best_i]], [best_acc], color='gold', s=200, marker='*', zorder=5,
               label=f'Best: {best_acc:.2f}%')
    ax.axhline(y=best_acc, color='gold', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Val Accuracy (Best: {best_acc:.2f}% @ Epoch {epochs[best_i]})')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if 'class_acc_history' in history and history['class_acc_history']:
        last = history['class_acc_history'][-1]
        cls  = list(last.keys())
        accs = [last[c] for c in cls]
        lbls = [class_names[c] if class_names and c < len(class_names) else f'C{c}' for c in cls]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cls)))
        bars = ax.bar(lbls, accs, color=colors)
        ax.set_xlabel('Class'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Class-wise Accuracy')
        ax.set_ylim(0, 100)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'Class accuracy\nnot available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('Class-wise Accuracy')

    if has_map and ncols == 4:
        ax = axes[1, 3]
        if 'per_class_map' in history and history['per_class_map']:
            pcm  = history['per_class_map'][-1]
            cids = sorted(pcm.keys())
            apv  = [pcm[c] * 100 for c in cids]
            lbls = [class_names[c] if class_names and c < len(class_names) else f'C{c}' for c in cids]
            clrs = plt.cm.RdYlGn(np.array(apv) / 100)
            bars = ax.bar(lbls, apv, color=clrs)
            ax.set_xlabel('Class'); ax.set_ylabel('AP50 (%)'); ax.set_title('Per-Class AP50')
            ax.set_ylim(0, 100)
            for bar, v in zip(bars, apv):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{v:.0f}', ha='center', va='bottom', fontsize=7)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Per-class AP\nnot available', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title('Per-Class AP50')

    plt.tight_layout()
    plt.savefig(save_dir / 'results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Results saved to {save_dir / 'results.png'}")


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """
    Plot confusion matrix with raw counts and normalized percentages.
    
    Args:
        cm: numpy array of shape (n_classes, n_classes)
        class_names: list of class name strings
        save_path: Path to save the figure
        title: Title for the figure
    """
    n_classes = len(class_names)
    
    # Create short names for display if names are too long
    short_names = []
    for name in class_names:
        if len(name) > 8:
            # Abbreviate: take first 4 chars + last 3 chars
            short_names.append(name[:4] + '..' + name[-2:] if len(name) > 8 else name)
        else:
            short_names.append(name)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Left panel: Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks(range(n_classes))
    ax1.set_yticks(range(n_classes))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(short_names, fontsize=9)
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title('Raw Counts', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            if val > 0:
                color = 'white' if val > cm.max() * 0.5 else 'black'
                ax1.text(j, i, str(val), ha='center', va='center', color=color, fontsize=8)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Count', fontsize=10)
    
    # Right panel: Normalized (per-class accuracy)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_norm = np.nan_to_num(cm_norm)
    
    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Actual', fontsize=11)
    ax2.set_title('Normalized (%)', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            if val > 0.5:
                color = 'white' if val > 50 else 'black'
                ax2.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Accuracy (%)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ============================================================================
# Model Builders
# ============================================================================

def build_classification_model(model_name, num_classes, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    img_size = model_kwargs.pop('img_size', 224)
    
    # =========================================================================
    # SAAD-Net (Full - with CDL)
    # =========================================================================
    if model_name in ['saad_net', 'defect_lock_v2_imp']:
        if not DEFECT_LOCK_V2_IMP_AVAILABLE: 
            raise ImportError("defect_lock_v2_improved.py not found")
        config = model_kwargs.pop('config', 'default')
        return get_defect_lock_v2_improved_for_training(num_classes=num_classes, config=config, **model_kwargs)
    
    # =========================================================================
    # SAAD-Net without CDL (attention_only config)
    # =========================================================================
    if model_name == 'saad_net_no_cdl':
        if not DEFECT_LOCK_V2_IMP_AVAILABLE: 
            raise ImportError("defect_lock_v2_improved.py not found")
        return get_defect_lock_v2_improved_for_training(num_classes=num_classes, config='attention_only', **model_kwargs)
    
    # =========================================================================
    # ResNet Family (torchvision)
    # =========================================================================
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    if model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    # =========================================================================
    # EfficientNet Family (torchvision)
    # =========================================================================
    if model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    if model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    # =========================================================================
    # ConvNeXt Family (torchvision)
    # =========================================================================
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
    
    # =========================================================================
    # Swin Transformer V2 (torchvision)
    # =========================================================================
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
    
    # =========================================================================
    # Vision Transformer (timm - flexible image size)
    # =========================================================================
    if model_name == 'vit_base':
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_base_patch16_224', pretrained=True, 
                                       num_classes=num_classes, img_size=img_size)
            return model
        else:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
    
    if model_name == 'vit_large':
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_large_patch16_224', pretrained=True,
                                       num_classes=num_classes, img_size=img_size)
            return model
        else:
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
    
    if model_name == 'vit_small':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for ViT-Small")
        model = timm.create_model('vit_small_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    # =========================================================================
    # DeiT (Data-efficient Image Transformer) - Facebook
    # =========================================================================
    if model_name == 'deit_small':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for DeiT")
        model = timm.create_model('deit_small_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    if model_name == 'deit_base':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for DeiT")
        model = timm.create_model('deit_base_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    if model_name == 'deit3_small':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for DeiT3")
        model = timm.create_model('deit3_small_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    if model_name == 'deit3_base':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for DeiT3")
        model = timm.create_model('deit3_base_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    # =========================================================================
    # BEiT (BERT Pre-training of Image Transformers) - Microsoft
    # =========================================================================
    if model_name == 'beit_base':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for BEiT")
        model = timm.create_model('beit_base_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    if model_name == 'beit_large':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for BEiT")
        model = timm.create_model('beit_large_patch16_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    # =========================================================================
    # CaiT (Class-Attention in Image Transformers) - Facebook
    # =========================================================================
    if model_name == 'cait_s24':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for CaiT")
        model = timm.create_model('cait_s24_224', pretrained=True,
                                   num_classes=num_classes, img_size=img_size)
        return model
    
    # =========================================================================
    # PVTv2 (Pyramid Vision Transformer v2) - Multi-scale
    # =========================================================================
    if model_name == 'pvt_v2_b2':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for PVT")
        model = timm.create_model('pvt_v2_b2', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    if model_name == 'pvt_v2_b3':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for PVT")
        model = timm.create_model('pvt_v2_b3', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    if model_name == 'pvt_v2_b4':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for PVT")
        model = timm.create_model('pvt_v2_b4', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    # =========================================================================
    # Twins (Spatially Separable Self-Attention)
    # =========================================================================
    if model_name == 'twins_svt_small':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for Twins")
        model = timm.create_model('twins_svt_small', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    if model_name == 'twins_svt_base':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for Twins")
        model = timm.create_model('twins_svt_base', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    # =========================================================================
    # CrossViT (Multi-scale Vision Transformer)
    # =========================================================================
    if model_name == 'crossvit_small':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for CrossViT")
        model = timm.create_model('crossvit_small_240', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    if model_name == 'crossvit_base':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for CrossViT")
        model = timm.create_model('crossvit_base_240', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    # =========================================================================
    # PoolFormer (MetaFormer baseline - pooling instead of attention)
    # =========================================================================
    if model_name == 'poolformer_s24':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for PoolFormer")
        model = timm.create_model('poolformer_s24', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    if model_name == 'poolformer_s36':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for PoolFormer")
        model = timm.create_model('poolformer_s36', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    # =========================================================================
    # CAFormer (ConvNet + Attention Former) - MetaFormer
    # =========================================================================
    if model_name == 'caformer_s18':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for CAFormer")
        model = timm.create_model('caformer_s18', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    # =========================================================================
    # EfficientFormer (Mobile-friendly ViT)
    # =========================================================================
    if model_name == 'efficientformer_l1':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for EfficientFormer")
        model = timm.create_model('efficientformer_l1', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    if model_name == 'efficientformer_l3':
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for EfficientFormer")
        model = timm.create_model('efficientformer_l3', pretrained=True,
                                   num_classes=num_classes)
        return model
    
    # =========================================================================
    # MaxViT (torchvision)
    # =========================================================================
    if model_name == 'maxvit_t':
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)
        return model
    
    # =========================================================================
    # Legacy models (swin_v2_ultra, enhanced, etc.)
    # =========================================================================
    if model_name == 'swin_v2_ultra':
        if not ULTRA_AVAILABLE: raise ImportError("swin_v2_ultra_enhanced.py not found")
        config = model_kwargs.pop('config', 'default')
        return get_swin_v2_ultra(num_classes=num_classes, config=config, **model_kwargs)
    
    highres_models = ['dinov2', 'eva02', 'internimage', 'focalnet', 'convnext_v2']
    if model_name in highres_models:
        if not HIGHRES_CLASS_AVAILABLE: raise ImportError("highres_classification_models.py not found")
        return get_highres_classifier(model_name, num_classes, **model_kwargs)
    
    if model_name == 'swin_v2_enhanced':
        if not ENHANCED_AVAILABLE: raise ImportError("swin_v2_enhanc.py not found")
        return SwinV2EnhancedClassifier(num_classes=num_classes, pretrained=True, **model_kwargs)
    
    if model_name == 'swin_v2_arcface':
        if not ENHANCED_AVAILABLE: raise ImportError("swin_v2_enhanc.py not found")
        return SwinV2ArcFaceClassifier(num_classes=num_classes, pretrained=True, **model_kwargs)
    
    # =========================================================================
    # Fallback to transformer_classifiers
    # =========================================================================
    return get_transformer_classifier(model_name=model_name, num_classes=num_classes,
                                      pretrained=True, **model_kwargs)


def build_anomaly_model(model_name, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    highres_models = ['reverse_distillation', 'rd4ad', 'fastflow', 'draem']
    if model_name in highres_models:
        if not HIGHRES_ANOMALY_AVAILABLE: raise ImportError("highres_anomaly_models.py not found")
        return get_highres_anomaly_detector(model_name, **model_kwargs)
    if not ANOMALY_AVAILABLE: raise ImportError("anomaly_detection_models.py not found")
    return get_anomaly_detector(model_name, **model_kwargs)


def build_detection_model(model_name, num_classes, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    if model_name == 'saad_net_v2_detection':
        if not SAAD_NET_V2_AVAILABLE: raise ImportError("saad_net_v2.py not found.")
        config = model_kwargs.pop('saad_v2_config', 'v2_default')
        model_kwargs.pop('config', None)
        return get_saad_net_v2(num_classes=num_classes, config=config, **model_kwargs)
    if model_name == 'saad_net_detection':
        if not SAAD_NET_AVAILABLE: raise ImportError("saad_net.py not found.")
        config = model_kwargs.pop('config', 'saad_default')
        return get_saad_net(num_classes=num_classes, config=config, **model_kwargs)
    raise ValueError(f"Unknown detection model: {model_name}")


# ============================================================================
# Training Functions (Classification / Anomaly - unchanged)
# ============================================================================

def train_classification_epoch(model, loader, criterion, optimizer, device, scaler, use_arcface=False):
    model.train()
    total_loss = 0.0; correct = 0; total = 0
    pbar = tqdm(loader, desc="Training") if is_main_process() else loader
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, labels) if use_arcface else model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total


def train_classification_epoch_ultra(model, loader, criterion, optimizer, device, scaler,
                                     use_arcface=False, use_sam=False, use_mixup=False,
                                     use_cutmix=False, mixup_alpha=0.2, ema=None,
                                     current_contrastive_weight=None):
    model.train()
    total_loss = 0.0; correct = 0; total = 0
    if ULTRA_AVAILABLE:
        from swin_v2_ultra_enhanced import mixup_data, cutmix_data, mixup_criterion
    pbar = tqdm(loader, desc="Training") if is_main_process() else loader
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        use_mix = (use_mixup or use_cutmix) and ULTRA_AVAILABLE and np.random.rand() < 0.3
        if use_mix:
            if use_cutmix and (not use_mixup or np.random.rand() < 0.5):
                images, labels_a, labels_b, lam = cutmix_data(images, labels, mixup_alpha)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs['logits']
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    if hasattr(criterion, 'forward') and 'DefectLoCK' in criterion.__class__.__name__:
                        kw = {'contrastive_weight': current_contrastive_weight} if current_contrastive_weight is not None else {}
                        loss = criterion(outputs, labels, **kw)['total']
                    else:
                        loss = criterion(logits, labels)
                outputs = logits
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) if use_mix else criterion(outputs, labels)
            else:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam) if use_mix else criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        if ema is not None: ema.update()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels_a).sum().item()
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(loader), 100. * correct / total


def train_anomaly_epoch(model, loader, optimizer, device, scaler, model_name):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training") if is_main_process() else loader
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if model_name == 'efficientad':
                loss = model.compute_loss(images)['total']
            elif model_name in ['simplenet', 'stfpm', 'reverse_distillation', 'rd4ad', 'fastflow']:
                loss = model.compute_loss(images)
            elif model_name == 'draem':
                noise = torch.randn_like(images) * 0.1
                mask  = (torch.rand(images.size(0), 1, images.size(2), images.size(3), device=device) > 0.7).float()
                loss  = model.compute_loss(images, images + noise * mask, mask)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
        if is_main_process(): pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_classification(model, loader, criterion, device, return_confusion_matrix=False, num_classes=None):
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    class_correct: Dict[int, int] = {}; class_total: Dict[int, int] = {}
    all_preds = []  # For confusion matrix
    all_labels = []  # For confusion matrix
    pbar = tqdm(loader, desc="Evaluating") if is_main_process() else loader
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, dict):
            logits = outputs['logits']
            if hasattr(criterion, 'forward') and 'DefectLoCK' in criterion.__class__.__name__:
                loss = criterion(outputs, labels)['total']
            else:
                loss = criterion(logits, labels)
            outputs = logits
        elif isinstance(outputs, tuple):
            outputs = outputs[0]; loss = criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0); correct += predicted.eq(labels).sum().item()
        
        # Collect predictions and labels for confusion matrix
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            label = int(label)
            class_total[label]   = class_total.get(label, 0) + 1
            if pred == label:
                class_correct[label] = class_correct.get(label, 0) + 1
    class_acc = {k: 100. * class_correct.get(k, 0) / class_total[k] for k in class_total}
    
    if return_confusion_matrix:
        from sklearn.metrics import confusion_matrix as sklearn_cm
        
        # Infer num_classes if not provided
        if num_classes is None:
            num_classes = max(max(all_labels) + 1 if all_labels else 0, 
                            max(all_preds) + 1 if all_preds else 0)
        
        # Compute local confusion matrix
        if len(all_labels) > 0:
            local_cm = sklearn_cm(all_labels, all_preds, labels=list(range(num_classes)))
        else:
            local_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        # DDP: sum confusion matrices from all ranks (confusion matrices are additive)
        if dist.is_initialized():
            cm_tensor = torch.tensor(local_cm, dtype=torch.long, device=device)
            dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
            cm = cm_tensor.cpu().numpy()
        else:
            cm = local_cm
        
        return total_loss / len(loader), 100. * correct / total, class_acc, cm
    return total_loss / len(loader), 100. * correct / total, class_acc


@torch.no_grad()
def evaluate_anomaly(model, loader, device):
    model.eval()
    scores = []; labels = []
    for images, batch_labels in tqdm(loader, desc="Evaluating") if is_main_process() else loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        score  = output[0] if isinstance(output, tuple) else output
        scores.extend(score.cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().tolist())
    scores = np.array(scores); labels = np.array(labels)
    from sklearn.metrics import roc_auc_score, f1_score
    try:
        auroc = roc_auc_score(labels, scores)
    except:
        auroc = 0.5
    best_f1 = 0; best_thresh = 0.5
    for thresh in np.linspace(scores.min(), scores.max(), 100):
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thresh = thresh
    return {'auroc': auroc, 'best_f1': best_f1, 'best_threshold': best_thresh}


# ============================================================================
# mAP50 Evaluation Utilities
# ============================================================================

def _box_iou_numpy(boxes1, boxes2):
    """Vectorized IoU: (M,4) x (N,4) -> (M,N)."""
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = a1[:, None] + a2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def _nms_torch(boxes, iou_thresh):
    """
    Fast NMS using torchvision.ops.nms (CUDA C++ kernel).
    boxes: (N,6) np.ndarray [x1,y1,x2,y2,score,cls]
    """
    if len(boxes) == 0:
        return boxes
    try:
        from torchvision.ops import nms as tv_nms
        result = []
        for cls_id in np.unique(boxes[:, 5]):
            m   = boxes[:, 5] == cls_id
            cb  = boxes[m]
            b_t = torch.from_numpy(cb[:, :4]).float()
            s_t = torch.from_numpy(cb[:, 4]).float()
            keep = tv_nms(b_t, s_t, iou_thresh).numpy()
            result.append(cb[keep])
        return np.concatenate(result, axis=0) if result else np.zeros((0, 6), dtype=np.float32)
    except Exception:
        # fallback: pure numpy NMS
        result = []
        for cls_id in np.unique(boxes[:, 5]):
            m  = boxes[:, 5] == cls_id
            cb = boxes[m][np.argsort(-boxes[m][:, 4])]
            while len(cb) > 0:
                result.append(cb[0])
                if len(cb) == 1: break
                iou = _box_iou_numpy(cb[0:1, :4], cb[1:, :4])[0]
                cb  = cb[1:][iou <= iou_thresh]
        return np.stack(result, axis=0) if result else np.zeros((0, 6), dtype=np.float32)


def decode_fcos_predictions(det_outputs, img_size, score_thresh=0.001, nms_thresh=0.5,
                             pre_nms_topk=1000):
    """
    Decode FCOS detection head outputs to boxes.

    OPTIMIZED:
    - All ops on GPU until after score filter + top-k + NMS
    - pre_nms_topk: cap boxes per image before NMS (default 1000)
      Without this, epoch 2+ boxes explode as model learns -> CPU NMS chokes
    - GPU batched_nms for final dedup
    """
    from torchvision.ops import batched_nms

    img_h, img_w  = img_size
    device        = det_outputs[0]['cls_logits'].device
    B             = det_outputs[0]['cls_logits'].shape[0]

    batch_boxes = [[] for _ in range(B)]

    for level_out in det_outputs:
        cls_logits = level_out['cls_logits'].sigmoid()
        bbox_pred  = level_out['bbox_pred'].float()
        # BUG FIX: centerness is now raw logit (sigmoid removed from AnchorFreeHead)
        # Apply sigmoid exactly once here
        centerness = level_out['centerness'].float().sigmoid()
        stride     = float(level_out['stride'])
        B_, C, fH, fW = cls_logits.shape

        grid_y = (torch.arange(fH, device=device, dtype=torch.float32) + 0.5) * stride
        grid_x = (torch.arange(fW, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')

        scores_map = cls_logits * centerness
        max_scores, cls_ids = scores_map.max(dim=1)

        for b in range(B_):
            s_flat    = max_scores[b].reshape(-1)
            c_flat    = cls_ids[b].reshape(-1)
            ltrb_flat = bbox_pred[b].reshape(4, -1).T
            gx_flat   = grid_x.reshape(-1)
            gy_flat   = grid_y.reshape(-1)

            # Score threshold filter
            keep = s_flat > score_thresh
            if not keep.any():
                continue

            s    = s_flat[keep]
            c    = c_flat[keep].float()
            ltrb = ltrb_flat[keep]
            gx   = gx_flat[keep]
            gy   = gy_flat[keep]

            # Top-k per level (prevents box explosion in early training)
            if s.shape[0] > pre_nms_topk:
                topk_idx = s.topk(pre_nms_topk).indices
                s    = s[topk_idx]
                c    = c[topk_idx]
                ltrb = ltrb[topk_idx]
                gx   = gx[topk_idx]
                gy   = gy[topk_idx]

            # BUG FIX (PRIMARY): DO NOT multiply ltrb by stride.
            #
            # Training target (_fcos_assign_level in saad_net.py) stores l,t,r,b
            # as PIXEL-SPACE distances:
            #   l = grid_x - x1_px  (e.g., l=100px for a box 100px to the left)
            # The head applies torch.exp() -> outputs positive pixel distances.
            #
            # Previous code did: x1 = gx - ltrb[0] * stride
            # For P3 (stride=8): a 50px defect gives ltrb?25, then 25*8=200 -> 400px box
            # -> all small defects (punching_hole, crease) decoded 8-32x too large -> 0% AP
            #
            # waist_folding survived because its ~800px box decoded to full-image size,
            # still overlapping with GT (IoU?0.61 > 0.5 threshold).
            #
            # Fix: gx and gy are already in pixel space; ltrb is pixel distance -> no scaling.
            x1 = (gx - ltrb[:, 0]).clamp(0, img_w)
            y1 = (gy - ltrb[:, 1]).clamp(0, img_h)
            x2 = (gx + ltrb[:, 2]).clamp(0, img_w)
            y2 = (gy + ltrb[:, 3]).clamp(0, img_h)

            valid = (x2 > x1) & (y2 > y1)
            if not valid.any():
                continue

            boxes_b = torch.stack([x1[valid], y1[valid], x2[valid], y2[valid],
                                    s[valid], c[valid]], dim=1)
            batch_boxes[b].append(boxes_b)

    final_results = []
    for b in range(B):
        if not batch_boxes[b]:
            final_results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        boxes_all = torch.cat(batch_boxes[b], dim=0)

        # Global top-k across all levels (YOLO also does this)
        if boxes_all.shape[0] > pre_nms_topk * 2:
            topk_idx  = boxes_all[:, 4].topk(pre_nms_topk * 2).indices
            boxes_all = boxes_all[topk_idx]

        xyxy   = boxes_all[:, :4]
        scores = boxes_all[:, 4]
        labels = boxes_all[:, 5].long()

        keep   = batched_nms(xyxy, scores, labels, nms_thresh)
        result = boxes_all[keep].cpu().numpy().astype(np.float32)
        final_results.append(result)

    return final_results


def compute_ap_from_pr(tp_arr, fp_arr, n_gt):
    """Compute AP using 11-point interpolation from sorted TP/FP arrays."""
    if n_gt == 0: return 0.0
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)
    recall    = tp_cum / max(n_gt, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p / 11.0
    return ap


def compute_detection_stats(all_preds, all_gts, num_classes, img_size, iou_thresh=0.5,
                            ap50_only=True):
    """
    Compute per-class AP50 (and optionally AP50-95) using YOLO-style vectorized matching.

    Key speedups vs naive implementation:
    - IoU matrix is computed per-image (all preds x all GTs at once), not per-detection
    - Matching uses numpy ops instead of Python loops
    - GT stored once per image, not copied into each detection tuple
    """
    img_h, img_w = img_size

    # Convert GT: normalized (cls,cx,cy,w,h) -> pixel (cls,x1,y1,x2,y2)
    gt_px = []
    for gt in all_gts:
        if gt.shape[0] == 0:
            gt_px.append(np.zeros((0, 5), dtype=np.float32))
            continue
        cls = gt[:, 0]; cx = gt[:, 1]*img_w; cy = gt[:, 2]*img_h
        bw  = gt[:, 3]*img_w; bh = gt[:, 4]*img_h
        gt_px.append(np.stack([cls, cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], axis=1).astype(np.float32))

    # Count images / instances per class
    img_counts  = {}
    inst_counts = {}
    for gt in gt_px:
        seen = set()
        for row in gt:
            c = int(row[0])
            inst_counts[c] = inst_counts.get(c, 0) + 1
            seen.add(c)
        for c in seen:
            img_counts[c] = img_counts.get(c, 0) + 1

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    per_class_ap50 = {}
    per_class_ap95 = {}
    per_class_prec = {}
    per_class_rec  = {}

    # Pre-build per-image IoU matrices for all classes at once (YOLO trick)
    # iou_cache[img_i] = (pred_boxes, pred_scores, pred_cls, gt_boxes_per_cls)
    # -- instead we precompute per-class per-image data

    for c in range(num_classes):
        # Gather (score, img_i, det_idx_in_img) and per-image GT for class c
        # Store per-image: list of (pred_boxes_c, pred_scores_c) and gt_boxes_c
        img_preds_c = []   # per image: np.ndarray (Ni, 5) [x1,y1,x2,y2,score]
        img_gts_c   = []   # per image: np.ndarray (Gi, 4) [x1,y1,x2,y2]
        n_gt_total  = 0

        for i, (pred, gt) in enumerate(zip(all_preds, gt_px)):
            c_gt   = gt[gt[:, 0] == c, 1:5] if gt.shape[0] > 0 else np.zeros((0, 4), dtype=np.float32)
            c_pred = pred[pred[:, 5] == c, :5] if pred.shape[0] > 0 else np.zeros((0, 5), dtype=np.float32)
            img_preds_c.append(c_pred)
            img_gts_c.append(c_gt)
            n_gt_total += len(c_gt)

        if n_gt_total == 0:
            continue

        def _match_at_thresh(thr):
            """
            Vectorized greedy matching for all images.
            Inner loop over detections replaced with numpy advanced indexing.
            """
            all_scores = []
            all_tp     = []
            all_fp     = []

            for preds_i, gts_i in zip(img_preds_c, img_gts_c):
                if len(preds_i) == 0:
                    continue
                scores = preds_i[:, 4]
                boxes  = preds_i[:, :4]
                n_gt_i = len(gts_i)

                all_scores.append(scores)

                if n_gt_i == 0:
                    all_tp.append(np.zeros(len(preds_i), dtype=np.float32))
                    all_fp.append(np.ones(len(preds_i),  dtype=np.float32))
                    continue

                # Full IoU matrix: (n_pred, n_gt) - one call per image
                iou_mat  = _box_iou_numpy(boxes, gts_i)   # (n_pred, n_gt)
                order    = np.argsort(-scores)             # high -> low score

                tp_i     = np.zeros(len(preds_i), dtype=np.float32)
                fp_i     = np.ones(len(preds_i),  dtype=np.float32)
                gt_taken = np.zeros(n_gt_i, dtype=bool)

                # Vectorized greedy: process detections in score order
                # For each det, find best unmatched GT; if IoU >= thr, it's TP
                for di in order:
                    row  = iou_mat[di].copy()
                    row[gt_taken] = -1.0          # mask matched GTs
                    best = int(row.argmax())
                    if row[best] >= thr:
                        tp_i[di] = 1.0
                        fp_i[di] = 0.0
                        gt_taken[best] = True

                all_tp.append(tp_i)
                all_fp.append(fp_i)

            if not all_scores:
                return np.array([]), np.array([]), 0

            scores_all = np.concatenate(all_scores)
            tp_all     = np.concatenate(all_tp)
            fp_all     = np.concatenate(all_fp)

            order = np.argsort(-scores_all)
            return tp_all[order], fp_all[order], n_gt_total

        # AP50
        tp50, fp50, n_gt = _match_at_thresh(0.5)
        per_class_ap50[c] = compute_ap_from_pr(tp50, fp50, n_gt)

        # P/R at best F1
        if len(tp50) > 0:
            tp_cum = np.cumsum(tp50); fp_cum = np.cumsum(fp50)
            prec = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)
            rec  = tp_cum / max(n_gt_total, 1)
            f1   = 2 * prec * rec / np.clip(prec + rec, 1e-8, None)
            bi   = f1.argmax()
            per_class_prec[c] = float(prec[bi])
            per_class_rec[c]  = float(rec[bi])
        else:
            per_class_prec[c] = 0.0
            per_class_rec[c]  = 0.0

        # AP50-95 (only when requested)
        if ap50_only:
            per_class_ap95[c] = 0.0
            continue

        ap95_vals = []
        for thr in iou_thresholds:
            tp_t, fp_t, n_gt_t = _match_at_thresh(thr)
            ap95_vals.append(compute_ap_from_pr(tp_t, fp_t, n_gt_t))
        per_class_ap95[c] = float(np.mean(ap95_vals))

    return (per_class_ap50, per_class_ap95,
            per_class_prec, per_class_rec,
            img_counts, inst_counts)



# ============================================================================
# Detection Training & Evaluation
# ============================================================================

def print_yolo_table(epoch, epochs, class_names, map50, map50_95,
                     per_class_ap, per_class_ap95,
                     img_counts, inst_counts,
                     per_class_precision, per_class_recall,
                     is_final=False):
    """
    Print YOLO-style evaluation table.

    Example output:
                     Class   Images  Instances   P        R    mAP50  mAP50-95
                       all      689        632   0.601   0.551   0.541   0.271
              punching_hole       70        107   0.524   0.556   0.502   0.285
    """
    num_cls = len(class_names)

    # Header
    if is_final:
        print(f"\n{'='*90}")
        print(f"Final Results (best checkpoint)")
        print(f"{'='*90}")
    else:
        ep_str = f"Epoch {epoch}/{epochs}" if epoch is not None else ""
        print(f"\n{ep_str:>20s}  {'Class':>20s}  {'Images':>7s}  {'Instances':>9s}  "
              f"{'P':>7s}  {'R':>7s}  {'mAP50':>7s}  {'mAP50-95':>9s}")
        print("-" * 90)

    # Aggregate
    total_imgs  = sum(img_counts.values())  if img_counts  else 0
    total_insts = sum(inst_counts.values()) if inst_counts else 0
    mean_p      = float(np.mean([per_class_precision.get(c, 0) for c in range(num_cls) if c in per_class_ap])) if per_class_ap else 0.0
    mean_r      = float(np.mean([per_class_recall.get(c, 0)    for c in range(num_cls) if c in per_class_ap])) if per_class_ap else 0.0

    if is_final:
        print(f"{'':>20s}  {'all':>20s}  {total_imgs:>7d}  {total_insts:>9d}  "
              f"{mean_p:>7.3f}  {mean_r:>7.3f}  {map50:>7.3f}  {map50_95:>9.3f}")
    else:
        print(f"{'':>20s}  {'all':>20s}  {total_imgs:>7d}  {total_insts:>9d}  "
              f"{mean_p:>7.3f}  {mean_r:>7.3f}  {map50:>7.3f}  {map50_95:>9.3f}")

    # Per-class rows
    for c in range(num_cls):
        if c not in per_class_ap:
            continue
        cn   = class_names[c] if c < len(class_names) else f'cls_{c}'
        imgs = img_counts.get(c, 0)
        inst = inst_counts.get(c, 0)
        p    = per_class_precision.get(c, 0.0)
        r    = per_class_recall.get(c, 0.0)
        ap   = per_class_ap.get(c, 0.0)
        ap95 = per_class_ap95.get(c, 0.0)
        print(f"{'':>20s}  {cn:>20s}  {imgs:>7d}  {inst:>9d}  "
              f"{p:>7.3f}  {r:>7.3f}  {ap:>7.3f}  {ap95:>9.3f}")

    if is_final:
        print(f"{'='*90}\n")


def train_detection_epoch(model, loader, criterion, optimizer, device, scaler,
                          ema=None, current_pca_weight=None):
    model.train()
    total_loss = 0.0; total_det = 0.0; correct = 0; total = 0
    pbar = tqdm(loader, desc="Training[Det]") if is_main_process() else loader
    for images, cls_labels, boxes_list, _ in pbar:
        images     = images.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)
        boxes_list = [b.to(device) for b in boxes_list]
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # Pass cls_labels so Proto-PCA can update prototypes during training
            outputs   = model(images, cls_labels)
            kw        = {'pca_weight': current_pca_weight} if current_pca_weight is not None else {}
            loss_dict = criterion(outputs=outputs, targets=cls_labels, boxes_list=boxes_list, **kw)
            loss      = loss_dict['total']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer); scaler.update()
        # NOTE: DDP handles gradient sync automatically; no barrier needed here
        if ema is not None: ema.update()
        total_loss += loss.item()
        total_det  += (loss_dict.get('det_cls', torch.tensor(0.)).item()
                       + loss_dict.get('det_bbox', torch.tensor(0.)).item())
        logits    = outputs['logits'] if isinstance(outputs, dict) else outputs
        predicted = logits.argmax(1)
        total     += cls_labels.size(0); correct += predicted.eq(cls_labels).sum().item()
        if is_main_process():
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc':  f'{100.*correct/total:.2f}%',
                              'det':  f'{total_det/(pbar.n+1):.4f}'})
    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate_detection(model, loader, criterion, device,
                       iou_thresh=0.5, score_thresh=0.001, compute_map=True,
                       ap50_only=True):
    """
    Evaluate detection model.

    Optimizations vs naive:
    - @torch.no_grad() prevents gradient graph
    - criterion() skipped in eval (saves FCOS target assignment per batch)
    - dist.all_gather done on CPU (no GPU memory for pickle buffer)
    - _match_at_thresh inner loop vectorized with numpy advanced indexing

    Returns:
        val_loss, val_acc, class_acc, map50, map50_95,
        per_class_ap, per_class_ap95, per_class_prec, per_class_rec,
        img_counts, inst_counts
    """
    model.eval()
    total_loss = 0.0; correct = 0; total = 0
    class_correct: Dict[int, int] = {}; class_total: Dict[int, int] = {}

    all_preds: List[np.ndarray] = []
    all_gts:   List[np.ndarray] = []
    img_size_ref = None
    last_logits  = None
    num_batches  = 0

    pbar = tqdm(loader, desc="Evaluating[Det]") if is_main_process() else loader

    for images, cls_labels, boxes_list, _ in pbar:
        images         = images.to(device, non_blocking=True)
        cls_labels     = cls_labels.to(device, non_blocking=True)
        boxes_list_dev = [b.to(device) for b in boxes_list]

        outputs = model(images)  # no labels needed (Proto-PCA skips in eval mode)

        # --- val_loss: cheap focal loss only, skip expensive FCOS assignment ---
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        with torch.cuda.amp.autocast():
            loss_val = F.cross_entropy(logits, cls_labels)
        total_loss += loss_val.item()
        num_batches += 1

        last_logits = logits
        predicted   = logits.argmax(1)
        total      += cls_labels.size(0)
        correct    += predicted.eq(cls_labels).sum().item()

        for pred, label in zip(predicted.cpu().tolist(), cls_labels.cpu().tolist()):
            class_total[label]   = class_total.get(label, 0) + 1
            if pred == label:
                class_correct[label] = class_correct.get(label, 0) + 1

        # Collect predictions & GT for mAP
        if compute_map and isinstance(outputs, dict) and 'det_outputs' in outputs and outputs['det_outputs']:
            if img_size_ref is None:
                img_size_ref = (images.shape[2], images.shape[3])
            det_preds = decode_fcos_predictions(
                outputs['det_outputs'], img_size_ref,
                score_thresh=score_thresh, nms_thresh=iou_thresh)
            for b in range(len(det_preds)):
                all_preds.append(det_preds[b])
                gt_b = boxes_list[b].numpy() if boxes_list[b].shape[0] > 0 else np.zeros((0, 5), np.float32)
                all_gts.append(gt_b)

    # --- Sync loss/acc across ranks ---
    distributed = dist.is_initialized()
    if distributed:
        t = torch.tensor([total_loss, correct, total, num_batches],
                         dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, correct, total, num_batches = t.tolist()

        # --- CPU gather: avoids large GPU buffer allocation ---
        if compute_map:
            import pickle
            local_obj = (all_preds, all_gts)
            # broadcast_object_list works on CPU (no GPU memory needed)
            gather_list = [None] * dist.get_world_size() if is_main_process() else None
            dist.gather_object(local_obj, gather_list, dst=0)

            if is_main_process():
                all_preds = []; all_gts = []
                for preds_i, gts_i in gather_list:
                    all_preds.extend(preds_i)
                    all_gts.extend(gts_i)
            else:
                all_preds = []; all_gts = []

    class_acc = {k: 100. * class_correct.get(k, 0) / class_total[k] for k in class_total}

    # --- Compute mAP (rank 0 only) ---
    map50 = 0.0; map50_95 = 0.0; per_class_ap = {}
    per_class_ap95 = {}; per_class_prec = {}; per_class_rec = {}
    img_counts = {}; inst_counts = {}

    if is_main_process() and compute_map and all_preds and img_size_ref is not None and last_logits is not None:
        num_classes = last_logits.shape[1]
        (per_class_ap, per_class_ap95,
         per_class_prec, per_class_rec,
         img_counts, inst_counts) = compute_detection_stats(
            all_preds, all_gts, num_classes, img_size_ref, iou_thresh,
            ap50_only=ap50_only)
        map50    = float(np.mean(list(per_class_ap.values())))    if per_class_ap    else 0.0
        map50_95 = float(np.mean(list(per_class_ap95.values()))) if per_class_ap95 else 0.0

    return (total_loss / max(num_batches, 1), 100. * correct / max(total, 1), class_acc,
            map50, map50_95, per_class_ap, per_class_ap95,
            per_class_prec, per_class_rec, img_counts, inst_counts)


# ============================================================================
# Main Training Loops
# ============================================================================

def train_classification(args):
    distributed = dist.is_initialized()
    rank = get_rank(); world_size = get_world_size()
    device = torch.device(f'cuda:{rank}' if distributed else 'cuda')
    use_arcface = args.model in ['swin_v2_arcface'] or (args.model == 'swin_v2_ultra' and args.ultra_config == 'arcface')

    print_rank0(f"\n{'='*70}\nClassification Training: {args.model}\n{'='*70}")
    print_rank0(f"Image size: {args.img_size}")

    train_transform = get_classification_transforms(args.img_size, is_train=True)
    val_transform   = get_classification_transforms(args.img_size, is_train=False)
    train_dataset   = ClassificationDataset(os.path.join(args.data_dir, 'train'), train_transform)
    val_dataset     = ClassificationDataset(os.path.join(args.data_dir, 'val'),   val_transform)
    class_names     = train_dataset.classes
    
    # Auto-detect num_classes
    if args.num_classes == 4 and len(class_names) != 4:
        args.num_classes = len(class_names)
        print_rank0(f"Auto-detected num_classes: {args.num_classes}")

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                               sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               sampler=val_sampler, num_workers=4, pin_memory=True)

    # Build model_kwargs with img_size for ViT
    model_kwargs = {'model_size': args.model_size, 'img_size': args.img_size}
    
    if args.model == 'swin_v2_enhanced':   
        model_kwargs.update({'use_cbam': True, 'use_local_enhance': True})
    elif args.model == 'swin_v2_arcface':  
        model_kwargs.update({'arcface_s': 30.0, 'arcface_m': 0.5})
    elif args.model == 'swin_v2_ultra':    
        model_kwargs = {'model_size': args.model_size, 'config': args.ultra_config, 
                        'dropout': 0.3, 'drop_path_rate': 0.2, 'img_size': args.img_size}
    elif args.model in ['defect_lock_v2_imp', 'saad_net']: 
        model_kwargs = {'model_size': args.model_size, 'config': args.defect_v2_config, 
                        'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'saad_net_no_cdl':
        model_kwargs = {'model_size': args.model_size, 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'dinov2':           
        model_kwargs = {'model_size': 'base', 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'eva02':            
        model_kwargs = {'model_size': 'base', 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'internimage':      
        model_kwargs = {'model_size': 'small', 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'focalnet':         
        model_kwargs = {'model_size': 'base', 'pretrained': True, 'img_size': args.img_size}
    elif args.model == 'convnext_v2':      
        model_kwargs = {'model_size': 'base', 'pretrained': True, 'img_size': args.img_size}

    model = build_classification_model(args.model, args.num_classes, model_kwargs)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print_rank0(f"Parameters: {num_params:.2f}M")
    
    if distributed:
        if args.model not in ['defect_lock_v2_imp', 'saad_net', 'saad_net_no_cdl']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        use_find_unused = args.model in ['defect_lock_v2_imp', 'saad_net', 'saad_net_no_cdl',
                                          'swin_v2', 'vit_attention_pool', 'maxvit', 'efficient_vit',
                                          'vit_base', 'vit_large']
        model = DDP(model, device_ids=[rank], find_unused_parameters=use_find_unused)

    if args.model in ['defect_lock_v2_imp', 'saad_net'] and DEFECT_LOCK_V2_IMP_AVAILABLE:
        criterion = DefectLoCKv2ImprovedLoss(num_classes=args.num_classes, focal_gamma=2.0,
                                              label_smoothing=0.1, contrastive_weight=args.defect_lock_contrastive_weight, temperature=0.07)
    else:
        criterion = FocalLoss(gamma=2.0)

    if args.model in ['dinov2', 'eva02']:
        backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
        for p in backbone.parameters(): p.requires_grad = False
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr * 0.1 * world_size, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * world_size, weight_decay=0.05)

    ema = None
    if args.use_ema and ULTRA_AVAILABLE:
        ema = EMA(model.module if distributed else model, decay=0.9999)

    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    scaler    = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='max')
    save_dir  = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0
    history  = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'class_acc_history': []}

    # CRITICAL: Stop signal tensor for broadcasting early stopping to all ranks
    stop_flag = torch.zeros(1, dtype=torch.int32, device=device)

    for epoch in range(args.epochs):
        if distributed: train_sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs} (lr: {lr:.2e})")

        current_contrastive_weight = None
        if args.model in ['defect_lock_v2_imp', 'saad_net'] and args.defect_lock_contrastive_weight > 0:
            current_contrastive_weight = args.defect_lock_contrastive_weight * min(epoch / 10, 1.0)

        train_loss, train_acc = train_classification_epoch_ultra(
            model=model, loader=train_loader, criterion=criterion, optimizer=optimizer,
            device=device, scaler=scaler, use_arcface=use_arcface,
            use_sam=args.use_sam and ULTRA_AVAILABLE, use_mixup=args.use_mixup and ULTRA_AVAILABLE,
            use_cutmix=args.use_cutmix and ULTRA_AVAILABLE, mixup_alpha=args.mixup_alpha,
            ema=ema, current_contrastive_weight=current_contrastive_weight)

        if ema is not None: ema.apply_shadow()
        val_loss, val_acc, class_acc = evaluate_classification(model, val_loader, criterion, device)
        if ema is not None: ema.restore()

        # Check if this is the best model (need to broadcast decision to all ranks for confusion matrix)
        is_best = torch.tensor([0], dtype=torch.int, device=device)
        if is_main_process():
            if val_acc > best_acc:
                is_best.fill_(1)
        if distributed:
            dist.broadcast(is_best, src=0)
        
        # All ranks participate in confusion matrix computation if best model
        confusion_mat = None
        if is_best.item() == 1:
            if ema is not None: ema.apply_shadow()
            _, _, _, confusion_mat = evaluate_classification(
                model, val_loader, criterion, device, return_confusion_matrix=True, num_classes=args.num_classes)
            if ema is not None: ema.restore()

        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
            print("Class Accuracies:")
            for cls_idx, acc in class_acc.items():
                print(f"  {class_names[cls_idx] if cls_idx < len(class_names) else f'class_{cls_idx}'}: {acc:.2f}%")
            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss);     history['val_acc'].append(val_acc)
            history['lr'].append(lr);                 history['class_acc_history'].append(class_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                
                # Apply EMA for model saving
                if ema is not None: ema.apply_shadow()
                model_state = model.module.state_dict() if distributed else model.state_dict()
                if ema is not None: ema.restore()
                
                # Save confusion matrix (already computed above by all ranks)
                if confusion_mat is not None:
                    np.save(save_dir / 'confusion_matrix.npy', confusion_mat)
                    print(f"Confusion matrix saved to {save_dir / 'confusion_matrix.npy'}")
                
                torch.save({'epoch': epoch, 'model_state_dict': model_state, 'val_acc': val_acc,
                            'class_names': class_names, 'model_name': args.model, 'img_size': args.img_size,
                            'num_classes': args.num_classes, 'params_M': num_params},
                           save_dir / 'best_model.pth')
                print(f"*** Best model saved! Acc: {val_acc:.2f}% ***")
            if early_stopping(val_acc, epoch):
                print(f"\n*** Early stopping @ epoch {epoch+1} ***")
                if distributed:
                    stop_flag.fill_(1)
            elif args.patience > 0:
                print(f"Early stopping: {early_stopping.counter}/{args.patience}")

        # CRITICAL: Broadcast stop signal to ALL ranks so every rank breaks together
        if distributed:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    if is_main_process():
        with open(save_dir / 'history.json', 'w') as f: json.dump(history, f, indent=2)
        plot_training_results(history, save_dir, class_names)
        
        # Plot confusion matrix if available
        cm_path = save_dir / 'confusion_matrix.npy'
        if cm_path.exists():
            cm = np.load(cm_path)
            plot_confusion_matrix(cm, class_names, save_dir / 'confusion_matrix.png',
                                  title='SAAD-Net Confusion Matrix')
        
        print(f"\n{'='*60}\nTraining done! Best: {best_acc:.2f}%\nSaved: {save_dir}\n{'='*60}")


def train_anomaly(args):
    distributed = dist.is_initialized()
    rank = get_rank(); world_size = get_world_size()
    device = torch.device(f'cuda:{rank}' if distributed else 'cuda')

    train_transform = get_anomaly_transforms(args.img_size, is_train=True)
    val_transform   = get_anomaly_transforms(args.img_size, is_train=False)
    train_dataset   = AnomalyDataset(os.path.join(args.data_dir, 'train'), train_transform, is_train=True,  normal_class=args.normal_class)
    val_dataset     = AnomalyDataset(os.path.join(args.data_dir, 'val'),   val_transform,   is_train=False, normal_class=args.normal_class)
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, num_workers=4, pin_memory=True)

    model_kwargs = {}
    if args.model == 'efficientad': model_kwargs = {'model_size': 'small', 'use_autoencoder': True}
    elif args.model == 'simplenet': model_kwargs = {'backbone': 'wide_resnet50_2'}
    elif args.model == 'stfpm':     model_kwargs = {'backbone': 'resnet18'}
    elif args.model == 'patchcore': model_kwargs = {'backbone': 'wide_resnet50_2', 'coreset_ratio': 0.01}
    elif args.model in ['reverse_distillation', 'rd4ad']: model_kwargs = {'backbone': 'wide_resnet50_2', 'layers': ['layer1', 'layer2', 'layer3']}
    elif args.model == 'fastflow': model_kwargs = {'backbone': 'wide_resnet50_2', 'layers': ['layer2', 'layer3'], 'flow_steps': 8}
    elif args.model == 'draem':    model_kwargs = {'base_channels': 64}

    model = build_anomaly_model(args.model, model_kwargs)
    model = model.to(device)

    if args.model == 'patchcore':
        model.fit(train_loader, device)
        metrics = evaluate_anomaly(model, val_loader, device)
        print_rank0(f"AUROC: {metrics['auroc']:.4f}, F1: {metrics['best_f1']:.4f}")
        if is_main_process():
            save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'memory_bank': model.memory_bank, 'model_name': args.model, 'metrics': metrics}, save_dir / 'patchcore_model.pth')
        return

    if distributed: model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if args.model in ['efficientad', 'simplenet']:
        (model.module if distributed else model).set_normalization_params(train_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * world_size, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    scaler    = torch.cuda.amp.GradScaler()
    save_dir  = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = 0
    history    = {'train_loss': [], 'auroc': [], 'f1': [], 'lr': []}

    for epoch in range(args.epochs):
        if distributed: train_sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs} (lr: {lr:.2e})")
        train_loss = train_anomaly_epoch(model, train_loader, optimizer, device, scaler, args.model)
        metrics    = evaluate_anomaly(model, val_loader, device)
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}")
            print(f"AUROC: {metrics['auroc']:.4f}, F1: {metrics['best_f1']:.4f}")
            history['train_loss'].append(train_loss); history['auroc'].append(metrics['auroc'])
            history['f1'].append(metrics['best_f1']); history['lr'].append(lr)
            if metrics['auroc'] > best_auroc:
                best_auroc = metrics['auroc']
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({'epoch': epoch, 'model_state_dict': model_state,
                            'auroc': metrics['auroc'], 'threshold': metrics['best_threshold'],
                            'model_name': args.model}, save_dir / 'best_model.pth')
                print(f"*** Best model saved! AUROC: {metrics['auroc']:.4f} ***")

    if is_main_process():
        with open(save_dir / 'history.json', 'w') as f: json.dump(history, f)
        print(f"\nTraining done! Best AUROC: {best_auroc:.4f}")


def train_detection(args):
    distributed = dist.is_initialized()
    rank = get_rank(); world_size = get_world_size()
    device = torch.device(f'cuda:{rank}' if distributed else 'cuda')
    _is_v2 = args.model == 'saad_net_v2_detection'

    print_rank0(f"\n{'='*70}")
    print_rank0(f"SAAD-Net{'V2' if _is_v2 else ''} Detection Training")
    print_rank0(f"  model       : {args.model}")
    print_rank0(f"  v2_config   : {args.saad_v2_config}" if _is_v2 else f"  saad_config : {args.saad_config}")
    print_rank0(f"  num_classes : {args.num_classes}")
    print_rank0(f"  img_size    : {args.img_size}")
    print_rank0(f"  det_weight  : {args.saad_det_weight}")
    print_rank0(f"  pca_weight  : {args.saad_pca_weight}")
    print_rank0(f"{'='*70}")

    voc_train = os.path.join(args.data_dir, 'train')
    voc_val   = os.path.join(args.data_dir, 'val')
    if not Path(voc_train).exists():
        raise FileNotFoundError(f"VOC train folder not found: {voc_train}")

    train_tf = get_detection_transforms(args.img_size, is_train=True)
    val_tf   = get_detection_transforms(args.img_size, is_train=False)
    train_dataset = DetectionDataset(voc_train, train_tf, use_hflip=True)
    val_dataset   = DetectionDataset(voc_val,   val_tf,   use_hflip=False)
    class_names   = train_dataset.classes

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler   = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                               sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=True,
                               collate_fn=detection_collate_fn)
    val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               sampler=val_sampler, num_workers=4, pin_memory=True,
                               collate_fn=detection_collate_fn)

    _model_kw = {'model_size': args.model_size, 'config': args.saad_config, 'pretrained': True}
    if _is_v2: _model_kw['saad_v2_config'] = args.saad_v2_config

    model = build_detection_model(args.model, num_classes=args.num_classes, model_kwargs=_model_kw)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if _is_v2 and SAAD_NET_V2_AVAILABLE:
        criterion = get_saad_net_v2_loss(num_classes=args.num_classes, lambda_det=args.saad_det_weight,
                                          lambda_pca=args.saad_pca_weight, focal_gamma=2.0, label_smoothing=0.1)
        print_rank0(f"SAADNetV2Loss  focal=1.0  det={args.saad_det_weight}  pca={args.saad_pca_weight}")
    else:
        criterion = get_saad_net_loss(num_classes=args.num_classes, lambda_det=args.saad_det_weight,
                                       lambda_pca=args.saad_pca_weight, focal_gamma=2.0, label_smoothing=0.1)
        print_rank0(f"SAADNetLoss    focal=1.0  det={args.saad_det_weight}  pca={args.saad_pca_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * world_size, weight_decay=0.05)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup, args.epochs)
    scaler    = torch.cuda.amp.GradScaler()

    ema = None
    if args.use_ema and ULTRA_AVAILABLE:
        ema = EMA(model.module if distributed else model, decay=0.9999)
        print_rank0("Using EMA (decay=0.9999)")

    # *** Early stopping based on mAP50 ***
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001, mode='max')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    best_map50 = 0.0; best_acc = 0.0
    best_per_class_ap   = {}; best_per_class_ap95 = {}
    best_per_class_prec = {}; best_per_class_rec  = {}
    best_img_counts     = {}; best_inst_counts    = {}
    # carry-forward last full eval results (for ap50-95 in best model)
    last_full_map50_95      = 0.0
    last_full_per_class_ap95 = {}
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'lr': [], 'class_acc_history': [],
               'map50': [], 'map50_95': [], 'per_class_map': []}          # <-- NEW

    # Shared stop flag tensor for broadcasting early-stop signal across ranks
    stop_flag = torch.zeros(1, dtype=torch.int32, device=device)

    for epoch in range(args.epochs):
        if distributed: train_sampler.set_epoch(epoch)
        lr = scheduler.step(epoch)
        print_rank0(f"\nEpoch {epoch+1}/{args.epochs}  (lr: {lr:.2e})")

        T_warmup  = 10
        cur_pca_w = args.saad_pca_weight * min(epoch / max(T_warmup, 1), 1.0)
        if is_main_process() and epoch < T_warmup:
            print(f"  PCA Progressive Warmup: lambda_pca = {cur_pca_w:.4f}")

        train_loss, train_acc = train_detection_epoch(
            model=model, loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, scaler=scaler,
            ema=ema, current_pca_weight=cur_pca_w)

        if ema is not None: ema.apply_shadow()

        (val_loss, val_acc, class_acc,
         map50, map50_95, per_class_ap, per_class_ap95,
         per_class_prec, per_class_rec,
         img_counts, inst_counts) = evaluate_detection(
            model, val_loader, criterion, device,
            iou_thresh=0.5, score_thresh=0.001,
            compute_map=True,
            ap50_only=((epoch + 1) % args.eval_full_freq != 0))

        if ema is not None: ema.restore()

        if is_main_process():
            # -- YOLO-style per-epoch table ----------------------------------
            mean_p = float(np.mean([per_class_prec.get(c,0) for c in per_class_ap])) if per_class_ap else 0.0
            mean_r = float(np.mean([per_class_rec.get(c,0)  for c in per_class_ap])) if per_class_ap else 0.0
            total_imgs  = sum(img_counts.values())  if img_counts  else 0
            total_insts = sum(inst_counts.values()) if inst_counts else 0
            is_full_eval = ((epoch + 1) % args.eval_full_freq == 0)
            if is_full_eval:
                last_full_map50_95       = map50_95
                last_full_per_class_ap95 = per_class_ap95
            map95_str = f'{map50_95:>9.3f}' if is_full_eval else f'{"   -":>9s}'

            header = (f"{'Class':>20s}  {'Images':>7s}  {'Instances':>9s}  "
                      f"{'Box(P':>7s}  {'R':>7s}  {'mAP50':>7s}  {'mAP50-95)':>9s}")
            bar    = "-" * 82
            if epoch == 0:
                print(f"\n{header}\n{bar}")

            print(f"{'all':>20s}  {total_imgs:>7d}  {total_insts:>9d}  "
                  f"{mean_p:>7.3f}  {mean_r:>7.3f}  {map50:>7.3f}  {map95_str}")

            # Train info line
            extra = f"  [full: mAP50-95={map50_95:.3f}]" if is_full_eval else ""
            print(f"  Epoch {epoch+1}/{args.epochs}  "
                  f"loss={val_loss:.4f}  acc={val_acc:.2f}%  "
                  f"best_mAP50={best_map50:.3f}{extra}")

            history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss);     history['val_acc'].append(val_acc)
            history['lr'].append(lr);                 history['class_acc_history'].append(class_acc)
            history['map50'].append(map50)
            history['map50_95'].append(last_full_map50_95)  # carry-forward last full eval
            history['per_class_map'].append({int(k): v for k, v in per_class_ap.items()})

            # *** Save best model based on mAP50 ***
            if map50 > best_map50:
                best_map50 = map50; best_acc = val_acc
                best_per_class_ap    = per_class_ap
                best_per_class_ap95  = last_full_per_class_ap95  # use last full eval
                best_per_class_prec  = per_class_prec
                best_per_class_rec   = per_class_rec
                best_img_counts      = img_counts
                best_inst_counts     = inst_counts
                if ema is not None: ema.apply_shadow()
                model_state = model.module.state_dict() if distributed else model.state_dict()
                torch.save({
                    'epoch'           : epoch,
                    'model_state_dict': model_state,
                    'val_acc'         : val_acc,
                    'map50'           : map50,
                    'map50_95'        : last_full_map50_95,  # carry-forward
                    'per_class_ap'    : {int(k): v for k, v in per_class_ap.items()},
                    'class_names'     : class_names,
                    'model_name'      : args.model,
                    'saad_v2_config'  : getattr(args, 'saad_v2_config', None),
                    'saad_config'     : args.saad_config,
                    'img_size'        : args.img_size,
                    'num_classes'     : args.num_classes,
                }, save_dir / 'best_model.pth')
                if ema is not None: ema.restore()
                print(f"  *** Best model saved!  mAP50: {map50:.3f}  mAP50-95: {map50_95:.3f} ***")

            # Save latest checkpoint
            model_state = model.module.state_dict() if distributed else model.state_dict()
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model_state,
                'optimizer_state' : optimizer.state_dict(),
                'val_acc'         : val_acc,
                'map50'           : map50,
            }, save_dir / 'last_model.pth')

            # *** Early stopping on mAP50 ***
            if early_stopping(map50, epoch):
                print(f"\n*** Early stopping @ epoch {epoch+1} ***")
                print(f"Best mAP50: {best_map50*100:.2f}% @ epoch {early_stopping.best_epoch+1}")
                if distributed:
                    stop_flag.fill_(1)
            elif args.patience > 0:
                print(f"  Early stopping: {early_stopping.counter}/{args.patience}")

        # Broadcast stop decision from rank 0 to all ranks so every rank breaks together
        if distributed:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    if is_main_process():
        with open(save_dir / 'history.json', 'w') as f: json.dump(history, f, indent=2)
        plot_training_results(history, save_dir, class_names)

        # -- YOLO-style Final Summary --------------------------------------
        best_mean_map50_95 = float(np.mean(list(best_per_class_ap95.values()))) if best_per_class_ap95 else 0.0
        print_yolo_table(
            epoch=None, epochs=args.epochs,
            class_names=class_names,
            map50=best_map50, map50_95=best_mean_map50_95,
            per_class_ap=best_per_class_ap, per_class_ap95=best_per_class_ap95,
            img_counts=best_img_counts, inst_counts=best_inst_counts,
            per_class_precision=best_per_class_prec, per_class_recall=best_per_class_rec,
            is_final=True)

        print(f"Results saved to: {save_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified Training Script')
    parser.add_argument('--task',       type=str, default='classification',
                        choices=['classification', 'anomaly', 'detection'])
    parser.add_argument('--data_dir',   type=str, required=True)
    parser.add_argument('--normal_class', type=str, default='good')
    parser.add_argument('--model', type=str, default='swin_v2_t',
                        choices=[
                            # Torchvision CNN models
                            'resnet50', 'resnet101',
                            'efficientnet_b4', 'efficientnet_v2_s',
                            'convnext_tiny', 'convnext_small', 'convnext_base',
                            # Torchvision Transformers
                            'swin_v2_t', 'swin_v2_s', 'swin_v2_b',
                            'maxvit_t',
                            # ViT variants (timm)
                            'vit_small', 'vit_base', 'vit_large',
                            # DeiT (Data-efficient ViT)
                            'deit_small', 'deit_base', 'deit3_small', 'deit3_base',
                            # BEiT (BERT-style pretrain)
                            'beit_base', 'beit_large',
                            # CaiT (Class-Attention)
                            'cait_s24',
                            # PVTv2 (Pyramid ViT)
                            'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4',
                            # Twins (Spatially Separable)
                            'twins_svt_small', 'twins_svt_base',
                            # CrossViT (Multi-scale)
                            'crossvit_small', 'crossvit_base',
                            # PoolFormer/CAFormer (MetaFormer)
                            'poolformer_s24', 'poolformer_s36', 'caformer_s18',
                            # EfficientFormer
                            'efficientformer_l1', 'efficientformer_l3',
                            # SAAD-Net variants
                            'saad_net', 'saad_net_no_cdl', 'defect_lock_v2_imp',
                            # Legacy models
                            'swin_v2', 'swin_v2_enhanced', 'swin_v2_arcface', 'swin_v2_ultra',
                            'maxvit', 'vit_attention_pool', 'efficient_vit',
                            'dinov2', 'eva02', 'internimage', 'focalnet', 'convnext_v2',
                            # Detection models
                            'saad_net_detection', 'saad_net_v2_detection',
                            # Anomaly models
                            'efficientad', 'simplenet', 'patchcore', 'stfpm',
                            'reverse_distillation', 'rd4ad', 'fastflow', 'draem'
                        ])
    parser.add_argument('--model_size', type=str, default='t', choices=['xt', 't', 's', 'b'])
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--ultra_config', type=str, default='default',
                        choices=['default', 'fpn', 'deep', 'arcface', 'full'])
    parser.add_argument('--defect_v2_config', type=str, default='default',
                        choices=['baseline', 'cbam_only', 'cbam_laa', 'attention_only',
                                 'default', 'light', 'frozen', 'full'])
    parser.add_argument('--defect_lock_contrastive_weight', type=float, default=0.1)
    parser.add_argument('--saad_config', type=str, default='saad_default',
                        choices=['baseline', 'cbam_laa', 'saad_fpn', 'saad_fpn_pca', 'saad_default'])
    parser.add_argument('--saad_v2_config', type=str, default='v2_default',
                        choices=['v2_baseline', 'v2_ms_laa', 'v2_ms_laa_fda',
                                 'v2_default', 'v2_no_det', 'v2_light'])
    parser.add_argument('--saad_det_weight', type=float, default=0.5)
    parser.add_argument('--saad_pca_weight', type=float, default=0.3)
    parser.add_argument('--img_size',   type=int,   default=1024)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--lr',         type=float, default=5e-5)
    parser.add_argument('--warmup',     type=int,   default=5)
    parser.add_argument('--use_sam',    action='store_true')
    parser.add_argument('--use_ema',    action='store_true')
    parser.add_argument('--use_mixup',  action='store_true')
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=1.0)
    parser.add_argument('--use_tta',    action='store_true')
    parser.add_argument('--patience',   type=int,   default=50)
    parser.add_argument('--min_delta',  type=float, default=0.001)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--save_dir',   type=str,   default='./checkpoints')
    parser.add_argument('--eval_full_freq', type=int, default=10,
                        help='Compute full AP50-95 every N epochs (default: 10). AP50 is computed every epoch.')
    parser.add_argument('--local_rank', type=int,   default=-1)

    args = parser.parse_args()

    if args.model in ['saad_net_detection', 'saad_net_v2_detection'] and args.task == 'classification':
        args.task = 'detection'
        print_rank0(f"[INFO] model={args.model} -> auto task=detection")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        setup_distributed(int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']))

    def signal_handler(sig, frame):
        print_rank0("\nInterrupted. Cleaning up...")
        cleanup_distributed(); sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if   args.task == 'classification': train_classification(args)
        elif args.task == 'detection':      train_detection(args)
        else:                               train_anomaly(args)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()