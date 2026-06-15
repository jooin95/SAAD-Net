#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import gc
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from scipy import stats

warnings.filterwarnings('ignore')

# Import model
try:
    from defect_lock_v2_improved import (
        DefectLoCKv2Improved, get_defect_lock_v2_improved
    )
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: defect_lock_v2_improved.py not found")


# ============================================================================
# 1. Dataset Classes
# ============================================================================

class ClassificationDataset(Dataset):
    """Standard classification dataset"""
    
    def __init__(self, root_dir, transform=None, 
                 extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
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
        return image, label, img_path


# ============================================================================
# 2. Metrics Calculation
# ============================================================================

class MetricsCalculator:
    """Comprehensive metrics calculation with confidence intervals"""
    
    @staticmethod
    def wilson_ci(correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Wilson Score Confidence Interval
        More accurate for small sample sizes than normal approximation
        """
        if total == 0:
            return 0.0, 0.0
        
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = correct / total
        
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        
        return max(0, center - margin), min(1, center + margin)
    
    @staticmethod
    def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, 
                     metric_fn, n_bootstrap: int = 1000,
                     confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap Confidence Interval
        Returns: (metric_value, ci_lower, ci_upper)
        """
        n_samples = len(y_true)
        bootstrap_scores = []
        
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrap_scores.append(score)
        
        bootstrap_scores = np.array(bootstrap_scores)
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_scores, alpha * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
        
        return metric_fn(y_true, y_pred), ci_lower, ci_upper
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: List[str] = None,
                           n_bootstrap: int = 1000) -> Dict:
        """Compute comprehensive metrics with confidence intervals"""
        
        n_classes = len(np.unique(y_true))
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(n_classes)]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Bootstrap CI for key metrics
        acc_val, acc_ci_low, acc_ci_high = MetricsCalculator.bootstrap_ci(
            y_true, y_pred, accuracy_score, n_bootstrap
        )
        
        bal_acc_val, bal_acc_ci_low, bal_acc_ci_high = MetricsCalculator.bootstrap_ci(
            y_true, y_pred, balanced_accuracy_score, n_bootstrap
        )
        
        macro_f1_fn = lambda y, p: f1_score(y, p, average='macro', zero_division=0)
        f1_val, f1_ci_low, f1_ci_high = MetricsCalculator.bootstrap_ci(
            y_true, y_pred, macro_f1_fn, n_bootstrap
        )
        
        # Wilson CI for accuracy
        correct = (y_true == y_pred).sum()
        total = len(y_true)
        wilson_low, wilson_high = MetricsCalculator.wilson_ci(correct, total)
        
        # Per-class accuracy with Wilson CI
        per_class_stats = []
        for i, cls_name in enumerate(class_names):
            mask = y_true == i
            if mask.sum() > 0:
                cls_correct = ((y_true == y_pred) & mask).sum()
                cls_total = mask.sum()
                cls_acc = cls_correct / cls_total
                w_low, w_high = MetricsCalculator.wilson_ci(cls_correct, cls_total)
                
                per_class_stats.append({
                    'class': cls_name,
                    'accuracy': cls_acc * 100,
                    'ci_low': w_low * 100,
                    'ci_high': w_high * 100,
                    'f1': per_class_f1[i],
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'support': int(cls_total)
                })
        
        return {
            'accuracy': accuracy * 100,
            'accuracy_ci': (wilson_low * 100, wilson_high * 100),
            'accuracy_bootstrap_ci': (acc_ci_low * 100, acc_ci_high * 100),
            'balanced_accuracy': balanced_acc * 100,
            'balanced_accuracy_ci': (bal_acc_ci_low * 100, bal_acc_ci_high * 100),
            'macro_f1': macro_f1,
            'macro_f1_ci': (f1_ci_low, f1_ci_high),
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'per_class_stats': per_class_stats,
            'confusion_matrix': cm.tolist(),
            'n_samples': total,
            'n_classes': n_classes
        }


# ============================================================================
# 3. Robustness Testing
# ============================================================================

class RobustnessTest:
    """Test model robustness against various perturbations"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def add_gaussian_noise(self, image: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise"""
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)
    
    def add_motion_blur(self, image: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """Apply motion blur using average pooling approximation"""
        # Simple horizontal motion blur
        kernel = torch.ones(1, 1, 1, kernel_size, device=self.device) / kernel_size
        
        # Apply to each channel
        blurred = []
        for c in range(image.shape[1]):
            ch = image[:, c:c+1, :, :]
            ch_blurred = F.conv2d(ch, kernel, padding=(0, kernel_size//2))
            blurred.append(ch_blurred)
        
        return torch.cat(blurred, dim=1)
    
    def adjust_brightness(self, image: torch.Tensor, factor: float = 0.3) -> torch.Tensor:
        """Adjust brightness randomly"""
        adjustment = (torch.rand(1, device=self.device) * 2 - 1) * factor
        return torch.clamp(image + adjustment, 0, 1)
    
    def jpeg_compression(self, image: torch.Tensor, quality: int = 50) -> torch.Tensor:
        """Simulate JPEG compression artifacts"""
        # Approximate JPEG artifacts with DCT-like frequency reduction
        # Using average pooling and upsampling
        h, w = image.shape[2], image.shape[3]
        block_size = max(2, (100 - quality) // 10)
        
        if block_size > 1:
            downsampled = F.avg_pool2d(image, block_size)
            compressed = F.interpolate(downsampled, size=(h, w), mode='bilinear', align_corners=False)
        else:
            compressed = image
        
        return compressed
    
    def get_perturbations(self) -> Dict[str, callable]:
        """Get all perturbation functions"""
        return {
            'clean': lambda x: x,
            'gaussian_noise_0.05': lambda x: self.add_gaussian_noise(x, 0.05),
            'gaussian_noise_0.1': lambda x: self.add_gaussian_noise(x, 0.1),
            'gaussian_noise_0.15': lambda x: self.add_gaussian_noise(x, 0.15),
            'motion_blur_5': lambda x: self.add_motion_blur(x, 5),
            'motion_blur_7': lambda x: self.add_motion_blur(x, 7),
            'motion_blur_9': lambda x: self.add_motion_blur(x, 9),
            'brightness_0.2': lambda x: self.adjust_brightness(x, 0.2),
            'brightness_0.3': lambda x: self.adjust_brightness(x, 0.3),
            'jpeg_70': lambda x: self.jpeg_compression(x, 70),
            'jpeg_50': lambda x: self.jpeg_compression(x, 50),
            'jpeg_30': lambda x: self.jpeg_compression(x, 30),
        }


# ============================================================================
# 4. LAA Visualization
# ============================================================================

class LAAVisualizer:
    """Visualize LAA anomaly maps"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.anomaly_maps = []
        self.features = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture anomaly maps"""
        if hasattr(self.model, 'model'):
            # Wrapper case
            model = self.model.model
        else:
            model = self.model
        
        if hasattr(model, 'anomaly_module'):
            def hook_fn(module, input, output):
                # output is (enhanced, anomaly_map)
                if isinstance(output, tuple) and len(output) == 2:
                    self.anomaly_maps.append(output[1].detach().cpu())
            
            model.anomaly_module.register_forward_hook(hook_fn)
    
    def visualize(self, image: torch.Tensor, save_path: str, 
                  class_name: str = None, pred_class: str = None):
        """Generate LAA visualization"""
        self.anomaly_maps = []
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0).to(self.device))
        
        if not self.anomaly_maps:
            print("Warning: No anomaly map captured. Check if LAA module is enabled.")
            return
        
        anomaly_map = self.anomaly_maps[0].squeeze().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Normalize for display
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Anomaly map
        im = axes[1].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('LAA Anomaly Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(image_np)
        # Resize anomaly map to image size
        anomaly_resized = np.array(Image.fromarray(anomaly_map).resize(
            (image_np.shape[1], image_np.shape[0]), Image.BILINEAR
        ))
        axes[2].imshow(anomaly_resized, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Thresholded regions
        threshold = 0.5
        axes[3].imshow(image_np)
        mask = anomaly_resized > threshold
        overlay = np.zeros((*image_np.shape[:2], 4))
        overlay[mask] = [1, 0, 0, 0.5]  # Red with alpha
        axes[3].imshow(overlay)
        axes[3].set_title(f'High Anomaly Regions (>{threshold})')
        axes[3].axis('off')
        
        # Title
        title = f"LAA Visualization"
        if class_name:
            title += f" | True: {class_name}"
        if pred_class:
            title += f" | Pred: {pred_class}"
        fig.suptitle(title, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return anomaly_map


# ============================================================================
# 5. Failure Case Analysis
# ============================================================================

class FailureCaseAnalyzer:
    """Analyze and visualize failure cases"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
    
    def analyze(self, dataloader: DataLoader, output_dir: Path, 
                max_failures_per_pair: int = 3) -> Dict:
        """Analyze all failure cases"""
        self.model.eval()
        
        failures = defaultdict(list)
        all_predictions = []
        all_labels = []
        all_paths = []
        all_confidences = []
        
        with torch.no_grad():
            for images, labels, paths in tqdm(dataloader, desc="Analyzing failures"):
                images = images.to(self.device)
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=1)
                confidences, predictions = probs.max(dim=1)
                
                for i in range(len(labels)):
                    pred = predictions[i].item()
                    true = labels[i].item()
                    conf = confidences[i].item()
                    path = paths[i]
                    
                    all_predictions.append(pred)
                    all_labels.append(true)
                    all_paths.append(path)
                    all_confidences.append(conf)
                    
                    if pred != true:
                        pair_key = (self.class_names[true], self.class_names[pred])
                        failures[pair_key].append({
                            'path': path,
                            'true_class': self.class_names[true],
                            'pred_class': self.class_names[pred],
                            'confidence': conf,
                            'true_prob': probs[i, true].item(),
                            'pred_prob': probs[i, pred].item()
                        })
        
        # Sort failures by confidence (high confidence failures are more interesting)
        for key in failures:
            failures[key] = sorted(failures[key], key=lambda x: -x['confidence'])
        
        # Create failure summary
        failure_summary = {
            'total_samples': len(all_labels),
            'total_failures': sum(len(v) for v in failures.values()),
            'failure_rate': sum(len(v) for v in failures.values()) / len(all_labels) * 100,
            'confusion_pairs': {}
        }
        
        for (true_cls, pred_cls), cases in failures.items():
            failure_summary['confusion_pairs'][f"{true_cls} -> {pred_cls}"] = {
                'count': len(cases),
                'avg_confidence': np.mean([c['confidence'] for c in cases]),
                'examples': cases[:max_failures_per_pair]
            }
        
        # Visualize top failure cases
        self._visualize_failures(failures, output_dir, max_failures_per_pair)
        
        return failure_summary
    
    def _visualize_failures(self, failures: Dict, output_dir: Path, 
                           max_per_pair: int = 3):
        """Visualize failure cases"""
        failure_dir = output_dir / 'failure_cases'
        failure_dir.mkdir(parents=True, exist_ok=True)
        
        for (true_cls, pred_cls), cases in failures.items():
            for i, case in enumerate(cases[:max_per_pair]):
                try:
                    img = Image.open(case['path']).convert('RGB')
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img)
                    ax.set_title(
                        f"True: {true_cls} | Pred: {pred_cls}\n"
                        f"Confidence: {case['confidence']:.3f} | "
                        f"True prob: {case['true_prob']:.3f}",
                        fontsize=10
                    )
                    ax.axis('off')
                    
                    save_name = f"{true_cls}_to_{pred_cls}_{i+1}.png"
                    plt.savefig(failure_dir / save_name, dpi=100, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Error visualizing {case['path']}: {e}")


# ============================================================================
# 6. Main Evaluation Function
# ============================================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, 
                   device: torch.device, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Run model evaluation and return predictions"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_paths = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = [None] * len(labels)
        
        images = images.to(device)
        outputs = model(images)
        
        if isinstance(outputs, dict):
            logits = outputs['logits']
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        _, predictions = logits.max(dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_paths.extend(paths)
    
    return np.array(all_labels), np.array(all_predictions), all_paths


def run_robustness_evaluation(model: nn.Module, dataloader: DataLoader,
                              device: torch.device, class_names: List[str]) -> Dict:
    """Run robustness evaluation with various perturbations"""
    model.eval()
    robustness_tester = RobustnessTest(device)
    perturbations = robustness_tester.get_perturbations()
    
    results = {}
    
    for perturb_name, perturb_fn in perturbations.items():
        print(f"\nTesting: {perturb_name}")
        
        all_predictions = []
        all_labels = []
        
        # Clear GPU cache before each perturbation test
        torch.cuda.empty_cache()
        gc.collect()
        
        pbar = tqdm(dataloader, desc=perturb_name)
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
            
            # Process one image at a time if batch_size > 1 and still OOM
            try:
                with torch.no_grad():
                    images = images.to(device)
                    perturbed = perturb_fn(images)
                    
                    outputs = model(perturbed)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    _, predictions = logits.max(dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.numpy())
                
                # Explicitly delete tensors and clear cache
                del images, perturbed, outputs, logits, predictions
                
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            # Clear cache every batch for large images
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        
        if len(y_true) > 0:
            results[perturb_name] = {
                'accuracy': accuracy_score(y_true, y_pred) * 100,
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
                'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
            }
        else:
            results[perturb_name] = {
                'accuracy': 0, 'balanced_accuracy': 0, 'macro_f1': 0
            }
        
        # Clear after each perturbation type
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


# ============================================================================
# 7. Multi-Run Evaluation (for standard deviation)
# ============================================================================

def multi_run_evaluation(checkpoint_dir: Path, data_dir: Path, 
                         device: torch.device, n_runs: int = 3,
                         img_size: int = 512) -> Dict:
    """
    Run evaluation multiple times with different seeds
    For computing mean ¡¾ std across multiple training runs
    """
    results_per_run = []
    
    # Look for multiple checkpoints or use single checkpoint with different seeds
    checkpoints = list(checkpoint_dir.glob("**/best_model*.pth"))
    
    if len(checkpoints) < n_runs:
        print(f"Warning: Found only {len(checkpoints)} checkpoints, using single checkpoint")
        checkpoints = checkpoints * n_runs
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i, ckpt_path in enumerate(checkpoints[:n_runs]):
        print(f"\n=== Run {i+1}/{n_runs} ===")
        print(f"Checkpoint: {ckpt_path}")
        
        # Set seed for reproducibility
        torch.manual_seed(42 + i * 100)
        np.random.seed(42 + i * 100)
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Get model config
        num_classes = checkpoint.get('num_classes', 7)
        config = checkpoint.get('defect_v2_config', 'default')
        class_names = checkpoint.get('class_names', [f"Class_{j}" for j in range(num_classes)])
        
        # Build model
        model = get_defect_lock_v2_improved(
            num_classes=num_classes,
            model_size='t',
            config=config
        )
        
        # Load weights
        state_dict = checkpoint['model_state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Create dataloader
        dataset = ClassificationDataset(data_dir / 'val', transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Evaluate
        y_true, y_pred, _ = evaluate_model(model, dataloader, device, class_names)
        metrics = MetricsCalculator.compute_all_metrics(y_true, y_pred, class_names)
        
        results_per_run.append(metrics)
    
    # Aggregate results
    aggregated = {
        'accuracy_mean': np.mean([r['accuracy'] for r in results_per_run]),
        'accuracy_std': np.std([r['accuracy'] for r in results_per_run]),
        'balanced_accuracy_mean': np.mean([r['balanced_accuracy'] for r in results_per_run]),
        'balanced_accuracy_std': np.std([r['balanced_accuracy'] for r in results_per_run]),
        'macro_f1_mean': np.mean([r['macro_f1'] for r in results_per_run]),
        'macro_f1_std': np.std([r['macro_f1'] for r in results_per_run]),
        'n_runs': n_runs,
        'per_run_results': results_per_run
    }
    
    return aggregated


# ============================================================================
# 8. Report Generation
# ============================================================================

def generate_report(metrics: Dict, robustness_results: Dict, 
                    failure_summary: Dict, output_dir: Path):
    """Generate comprehensive evaluation report"""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE EVALUATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # Overall Metrics
    report.append("## OVERALL METRICS")
    report.append("-" * 40)
    report.append(f"Accuracy: {metrics['accuracy']:.2f}%")
    report.append(f"  Wilson 95% CI: ({metrics['accuracy_ci'][0]:.2f}%, {metrics['accuracy_ci'][1]:.2f}%)")
    report.append(f"  Bootstrap 95% CI: ({metrics['accuracy_bootstrap_ci'][0]:.2f}%, {metrics['accuracy_bootstrap_ci'][1]:.2f}%)")
    report.append(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    report.append(f"  Bootstrap 95% CI: ({metrics['balanced_accuracy_ci'][0]:.2f}%, {metrics['balanced_accuracy_ci'][1]:.2f}%)")
    report.append(f"Macro F1: {metrics['macro_f1']:.4f}")
    report.append(f"  Bootstrap 95% CI: ({metrics['macro_f1_ci'][0]:.4f}, {metrics['macro_f1_ci'][1]:.4f})")
    report.append(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    report.append(f"Macro Precision: {metrics['macro_precision']:.4f}")
    report.append(f"Macro Recall: {metrics['macro_recall']:.4f}")
    report.append(f"Total Samples: {metrics['n_samples']}")
    report.append("")
    
    # Per-class metrics
    report.append("## PER-CLASS METRICS")
    report.append("-" * 40)
    report.append(f"{'Class':<15} {'Acc%':<8} {'95% CI':<18} {'F1':<8} {'Prec':<8} {'Rec':<8} {'N':<6}")
    report.append("-" * 80)
    
    for stat in metrics['per_class_stats']:
        report.append(
            f"{stat['class']:<15} "
            f"{stat['accuracy']:>6.2f}  "
            f"({stat['ci_low']:>5.1f}, {stat['ci_high']:>5.1f})  "
            f"{stat['f1']:>6.4f}  "
            f"{stat['precision']:>6.4f}  "
            f"{stat['recall']:>6.4f}  "
            f"{stat['support']:>5d}"
        )
    report.append("")
    
    # Robustness Results
    if robustness_results:
        report.append("## ROBUSTNESS TEST RESULTS")
        report.append("-" * 40)
        report.append(f"{'Perturbation':<25} {'Acc%':<10} {'Bal.Acc%':<12} {'MacroF1':<10} {'¥Ä Acc':<10}")
        report.append("-" * 70)
        
        clean_acc = robustness_results.get('clean', {}).get('accuracy', 0)
        
        for perturb, res in robustness_results.items():
            delta = res['accuracy'] - clean_acc
            report.append(
                f"{perturb:<25} "
                f"{res['accuracy']:>8.2f}  "
                f"{res['balanced_accuracy']:>10.2f}  "
                f"{res['macro_f1']:>8.4f}  "
                f"{delta:>+8.2f}"
            )
        report.append("")
    
    # Failure Analysis
    if failure_summary:
        report.append("## FAILURE CASE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total Failures: {failure_summary['total_failures']} / {failure_summary['total_samples']}")
        report.append(f"Failure Rate: {failure_summary['failure_rate']:.2f}%")
        report.append("")
        report.append("Top Confusion Pairs:")
        
        sorted_pairs = sorted(
            failure_summary['confusion_pairs'].items(),
            key=lambda x: -x[1]['count']
        )
        
        for pair, info in sorted_pairs[:10]:
            report.append(f"  {pair}: {info['count']} cases (avg conf: {info['avg_confidence']:.3f})")
        report.append("")
    
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_dir / "evaluation_report.txt", "w") as f:
        f.write(report_text)
    
    print(report_text)
    
    return report_text


# ============================================================================
# 9. Confusion Matrix Visualization
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         output_path: Path, normalize: bool = True):
    """Plot confusion matrix with both counts and normalized values"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=12)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 10. Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation for SAAD-Net')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation (use 1-2 for 1024x1024 images on single GPU)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples for CI')
    
    # Optional analyses
    parser.add_argument('--visualize_laa', action='store_true',
                        help='Generate LAA anomaly map visualizations')
    parser.add_argument('--robustness_test', action='store_true',
                        help='Run robustness tests')
    parser.add_argument('--failure_analysis', action='store_true',
                        help='Run failure case analysis')
    parser.add_argument('--multi_run', action='store_true',
                        help='Run multi-run evaluation for std computation')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs for multi-run evaluation')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get config from checkpoint
    num_classes = checkpoint.get('num_classes', 7)
    config = checkpoint.get('defect_v2_config', 'default')
    class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(num_classes)])
    
    print(f"Model config: {config}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Build model
    if not MODEL_AVAILABLE:
        print("Error: Model not available")
        return
    
    model = get_defect_lock_v2_improved(
        num_classes=num_classes,
        model_size='t',
        config=config
    )
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    # Remove 'model.' prefix if present (from wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    data_path = Path(args.data_dir)
    if (data_path / 'val').exists():
        val_path = data_path / 'val'
    elif (data_path / 'test').exists():
        val_path = data_path / 'test'
    else:
        val_path = data_path
    
    dataset = ClassificationDataset(val_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Update class names from dataset if available
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    
    # ==================== Main Evaluation ====================
    print("\n" + "="*60)
    print("RUNNING MAIN EVALUATION")
    print("="*60)
    
    y_true, y_pred, paths = evaluate_model(model, dataloader, device, class_names)
    metrics = MetricsCalculator.compute_all_metrics(
        y_true, y_pred, class_names, n_bootstrap=args.n_bootstrap
    )
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    
    # ==================== Robustness Test ====================
    robustness_results = {}
    if args.robustness_test:
        print("\n" + "="*60)
        print("RUNNING ROBUSTNESS TESTS")
        print("="*60)
        robustness_results = run_robustness_evaluation(
            model, dataloader, device, class_names
        )
    
    # ==================== Failure Analysis ====================
    failure_summary = {}
    if args.failure_analysis:
        print("\n" + "="*60)
        print("RUNNING FAILURE CASE ANALYSIS")
        print("="*60)
        analyzer = FailureCaseAnalyzer(model, device, class_names)
        failure_summary = analyzer.analyze(dataloader, output_dir)
    
    # ==================== LAA Visualization ====================
    if args.visualize_laa:
        print("\n" + "="*60)
        print("GENERATING LAA VISUALIZATIONS")
        print("="*60)
        
        laa_dir = output_dir / 'laa_visualizations'
        laa_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = LAAVisualizer(model, device)
        
        # Visualize a few samples from each class
        samples_per_class = 3
        class_samples = defaultdict(list)
        
        for i, (img_path, label) in enumerate(dataset.samples):
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append(i)
        
        for label, indices in class_samples.items():
            for idx in indices:
                image, _, img_path = dataset[idx]
                
                # Get prediction
                with torch.no_grad():
                    outputs = model(image.unsqueeze(0).to(device))
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    pred = logits.argmax(dim=1).item()
                
                save_name = f"{class_names[label]}_{idx}_pred_{class_names[pred]}.png"
                visualizer.visualize(
                    image, 
                    str(laa_dir / save_name),
                    class_name=class_names[label],
                    pred_class=class_names[pred]
                )
    
    # ==================== Generate Report ====================
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    generate_report(metrics, robustness_results, failure_summary, output_dir)
    
    # Save metrics as JSON
    metrics_json = {
        'accuracy': metrics['accuracy'],
        'accuracy_ci_wilson': metrics['accuracy_ci'],
        'accuracy_ci_bootstrap': metrics['accuracy_bootstrap_ci'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'balanced_accuracy_ci': metrics['balanced_accuracy_ci'],
        'macro_f1': metrics['macro_f1'],
        'macro_f1_ci': metrics['macro_f1_ci'],
        'weighted_f1': metrics['weighted_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'per_class_stats': metrics['per_class_stats'],
        'n_samples': metrics['n_samples'],
        'robustness': robustness_results,
        'failure_summary': {
            k: v for k, v in failure_summary.items() 
            if k != 'confusion_pairs'
        } if failure_summary else {}
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n? Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()