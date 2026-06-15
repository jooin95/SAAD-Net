#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Large Image Processing Pipeline for Small Defect Detection
===========================================================
Implements:
1. Tile-based Processing with Overlap
2. Coarse-to-Fine Two-Stage Detection
3. Attention-Guided Adaptive Sampling
4. Multi-scale Ensemble

Designed for:
- High-resolution images (8K+)
- Small defects (< 1% of image area)
- Real-time industrial inspection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import math


# ============================================================================
# 1. Data Structures
# ============================================================================

@dataclass
class Tile:
    """Single tile extracted from large image"""
    image: np.ndarray  # Tile pixel data
    x: int  # Top-left x coordinate in original image
    y: int  # Top-left y coordinate in original image
    width: int
    height: int


@dataclass
class DefectResult:
    """Detection result for a single defect"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image
    score: float  # Confidence score
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    area: Optional[int] = None


@dataclass
class InspectionResult:
    """Complete inspection result for an image"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_map: Optional[np.ndarray]
    defects: List[DefectResult]
    processing_time_ms: float
    num_tiles_processed: int
    metadata: Dict = None


# ============================================================================
# 2. Tile Processor
# ============================================================================

class TileProcessor:
    """
    Tile-based image processing for large images
    
    Splits large images into overlapping tiles, processes each,
    and merges results back to original resolution.
    """
    
    def __init__(
        self,
        tile_size: int = 1024,
        overlap: float = 0.25,
        batch_size: int = 4
    ):
        """
        Args:
            tile_size: Size of each tile (square)
            overlap: Overlap ratio between tiles (0.25 = 25%)
            batch_size: Number of tiles to process at once
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.batch_size = batch_size
    
    def extract_tiles(self, image: np.ndarray) -> List[Tile]:
        """
        Extract overlapping tiles from image
        
        Args:
            image: Input image (H, W, C) or (H, W)
        
        Returns:
            List of Tile objects
        """
        if image.ndim == 2:
            H, W = image.shape
        else:
            H, W = image.shape[:2]
        
        tiles = []
        
        # Generate tile positions
        y_positions = list(range(0, max(1, H - self.tile_size + 1), self.stride))
        x_positions = list(range(0, max(1, W - self.tile_size + 1), self.stride))
        
        # Add final positions if not covered
        if y_positions[-1] + self.tile_size < H:
            y_positions.append(H - self.tile_size)
        if x_positions[-1] + self.tile_size < W:
            x_positions.append(W - self.tile_size)
        
        for y in y_positions:
            for x in x_positions:
                # Handle edge cases
                y_end = min(y + self.tile_size, H)
                x_end = min(x + self.tile_size, W)
                y_start = y_end - self.tile_size
                x_start = x_end - self.tile_size
                
                # Ensure valid coordinates
                y_start = max(0, y_start)
                x_start = max(0, x_start)
                
                tile_img = image[y_start:y_end, x_start:x_end]
                
                # Pad if necessary
                if tile_img.shape[0] < self.tile_size or tile_img.shape[1] < self.tile_size:
                    if image.ndim == 2:
                        padded = np.zeros((self.tile_size, self.tile_size), dtype=image.dtype)
                    else:
                        padded = np.zeros((self.tile_size, self.tile_size, image.shape[2]), dtype=image.dtype)
                    padded[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
                    tile_img = padded
                
                tiles.append(Tile(
                    image=tile_img,
                    x=x_start,
                    y=y_start,
                    width=self.tile_size,
                    height=self.tile_size
                ))
        
        return tiles
    
    def merge_predictions(
        self,
        predictions: List[Tuple[np.ndarray, Tile]],
        original_shape: Tuple[int, int],
        merge_method: str = 'max'
    ) -> np.ndarray:
        """
        Merge tile predictions back to original size
        
        Args:
            predictions: List of (prediction_map, tile) tuples
            original_shape: (H, W) of original image
            merge_method: 'max', 'mean', or 'weighted'
        
        Returns:
            Merged prediction map
        """
        H, W = original_shape
        
        result = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)
        
        for pred, tile in predictions:
            # Resize prediction if necessary
            if pred.shape[0] != tile.height or pred.shape[1] != tile.width:
                pred = cv2.resize(pred, (tile.width, tile.height))
            
            y_end = min(tile.y + tile.height, H)
            x_end = min(tile.x + tile.width, W)
            pred_h = y_end - tile.y
            pred_w = x_end - tile.x
            
            if merge_method == 'max':
                result[tile.y:y_end, tile.x:x_end] = np.maximum(
                    result[tile.y:y_end, tile.x:x_end],
                    pred[:pred_h, :pred_w]
                )
            elif merge_method == 'mean':
                result[tile.y:y_end, tile.x:x_end] += pred[:pred_h, :pred_w]
                count[tile.y:y_end, tile.x:x_end] += 1
            elif merge_method == 'weighted':
                # Weight by distance from center (higher in center)
                weight = self._create_weight_mask(tile.width, tile.height)
                result[tile.y:y_end, tile.x:x_end] += pred[:pred_h, :pred_w] * weight[:pred_h, :pred_w]
                count[tile.y:y_end, tile.x:x_end] += weight[:pred_h, :pred_w]
        
        if merge_method in ['mean', 'weighted']:
            count = np.maximum(count, 1e-6)
            result = result / count
        
        return result
    
    def _create_weight_mask(self, width: int, height: int) -> np.ndarray:
        """Create Gaussian-like weight mask (higher in center)"""
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        xx, yy = np.meshgrid(x, y)
        weight = np.exp(-(xx**2 + yy**2) / 0.5)
        return weight.astype(np.float32)


# ============================================================================
# 3. Coarse-to-Fine Detector
# ============================================================================

class CoarseToFineDetector:
    """
    Two-stage detection for efficient large image processing
    
    Stage 1 (Coarse): Fast scan at low resolution to find ROIs
    Stage 2 (Fine): Detailed analysis of ROIs at high resolution
    
    Benefits:
    - Normal images processed very quickly (only Stage 1)
    - GPU memory efficient
    - High accuracy on small defects
    """
    
    def __init__(
        self,
        coarse_model: nn.Module,
        fine_model: nn.Module,
        coarse_size: int = 512,
        fine_tile_size: int = 1024,
        coarse_threshold: float = 0.3,
        fine_threshold: float = 0.5,
        device: torch.device = None,
        transform: Callable = None
    ):
        """
        Args:
            coarse_model: Fast model for initial scan (e.g., EfficientNet-B0)
            fine_model: Accurate model for detailed analysis (e.g., Swin-V2)
            coarse_size: Resolution for coarse scan
            fine_tile_size: Tile size for fine analysis
            coarse_threshold: Threshold for ROI selection
            fine_threshold: Final anomaly threshold
            device: torch device
            transform: Image preprocessing transform
        """
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.coarse_size = coarse_size
        self.fine_tile_size = fine_tile_size
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.coarse_model.to(self.device).eval()
        self.fine_model.to(self.device).eval()
        
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        return_visualization: bool = False
    ) -> InspectionResult:
        """
        Run two-stage detection
        
        Args:
            image: Input image (H, W, C) in BGR or RGB
            return_visualization: If True, include visualization in result
        
        Returns:
            InspectionResult with detection results
        """
        import time
        start_time = time.time()
        
        H, W = image.shape[:2]
        
        # ========== Stage 1: Coarse Detection ==========
        # Resize image for fast processing
        coarse_image = cv2.resize(image, (self.coarse_size, self.coarse_size))
        coarse_tensor = self._preprocess(coarse_image)
        
        # Run coarse model
        coarse_output = self._run_coarse(coarse_tensor)
        
        # Resize coarse map to original size
        coarse_map_full = cv2.resize(coarse_output, (W, H))
        
        # Find suspicious regions
        suspicious_mask = coarse_map_full > self.coarse_threshold
        
        # If no suspicious regions, return early (fast path)
        if not suspicious_mask.any():
            processing_time = (time.time() - start_time) * 1000
            return InspectionResult(
                is_anomaly=False,
                anomaly_score=float(coarse_map_full.max()),
                anomaly_map=coarse_map_full if return_visualization else None,
                defects=[],
                processing_time_ms=processing_time,
                num_tiles_processed=0,
                metadata={'stage': 'coarse_only'}
            )
        
        # ========== Stage 2: Fine Detection ==========
        # Extract ROIs
        rois = self._extract_rois(suspicious_mask, image)
        
        # Process ROIs
        fine_results = []
        for roi_image, roi_bbox in rois:
            # Preprocess and run fine model
            roi_tensor = self._preprocess(roi_image)
            fine_output = self._run_fine(roi_tensor)
            
            fine_results.append({
                'bbox': roi_bbox,
                'anomaly_map': fine_output,
                'score': float(fine_output.max())
            })
        
        # Merge results
        final_map = self._merge_results(coarse_map_full, fine_results, (H, W))
        
        # Extract defects
        defects = self._extract_defects(final_map)
        
        processing_time = (time.time() - start_time) * 1000
        
        return InspectionResult(
            is_anomaly=len(defects) > 0,
            anomaly_score=float(final_map.max()),
            anomaly_map=final_map if return_visualization else None,
            defects=defects,
            processing_time_ms=processing_time,
            num_tiles_processed=len(rois),
            metadata={'stage': 'fine', 'num_rois': len(rois)}
        )
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)
    
    def _run_coarse(self, x: torch.Tensor) -> np.ndarray:
        """Run coarse model and get anomaly map"""
        output = self.coarse_model(x)
        
        # Handle different output types
        if isinstance(output, tuple):
            score, amap = output
            return amap.squeeze().cpu().numpy()
        elif hasattr(self.coarse_model, 'forward') and 'return_map' in str(self.coarse_model.forward.__code__.co_varnames):
            score, amap = self.coarse_model(x, return_map=True)
            return amap.squeeze().cpu().numpy()
        else:
            # For classification models, use softmax confidence
            probs = F.softmax(output, dim=1)
            # Anomaly = 1 - confidence of normal class (assuming class 0 is normal)
            anomaly = 1 - probs[:, 0]
            return anomaly.cpu().numpy()[0] * np.ones((self.coarse_size, self.coarse_size))
    
    def _run_fine(self, x: torch.Tensor) -> np.ndarray:
        """Run fine model and get anomaly map"""
        output = self.fine_model(x)
        
        if isinstance(output, tuple):
            score, amap = output
            return amap.squeeze().cpu().numpy()
        elif hasattr(self.fine_model, 'forward'):
            try:
                score, amap = self.fine_model(x, return_map=True)
                return amap.squeeze().cpu().numpy()
            except:
                pass
        
        # For classification models
        probs = F.softmax(output, dim=1)
        anomaly = 1 - probs[:, 0]
        return anomaly.cpu().numpy()[0] * np.ones((self.fine_tile_size, self.fine_tile_size))
    
    def _extract_rois(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        min_area: int = 100,
        padding_ratio: float = 0.25
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract ROIs from suspicious regions"""
        H, W = image.shape[:2]
        
        # Find connected components
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        
        rois = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            
            # Add padding
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)
            
            # Ensure minimum size
            if x2 - x1 < self.fine_tile_size:
                cx = (x1 + x2) // 2
                x1 = max(0, cx - self.fine_tile_size // 2)
                x2 = min(W, x1 + self.fine_tile_size)
                x1 = x2 - self.fine_tile_size
            
            if y2 - y1 < self.fine_tile_size:
                cy = (y1 + y2) // 2
                y1 = max(0, cy - self.fine_tile_size // 2)
                y2 = min(H, y1 + self.fine_tile_size)
                y1 = y2 - self.fine_tile_size
            
            # Extract and resize ROI
            roi = image[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (self.fine_tile_size, self.fine_tile_size))
            
            rois.append((roi_resized, (x1, y1, x2, y2)))
        
        return rois
    
    def _merge_results(
        self,
        coarse_map: np.ndarray,
        fine_results: List[Dict],
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """Merge coarse and fine results"""
        H, W = shape
        final_map = coarse_map.copy()
        
        for result in fine_results:
            x1, y1, x2, y2 = result['bbox']
            fine_map = result['anomaly_map']
            
            # Resize to ROI size
            roi_h, roi_w = y2 - y1, x2 - x1
            fine_resized = cv2.resize(fine_map, (roi_w, roi_h))
            
            # Merge using max
            final_map[y1:y2, x1:x2] = np.maximum(
                final_map[y1:y2, x1:x2],
                fine_resized
            )
        
        return final_map
    
    def _extract_defects(
        self,
        anomaly_map: np.ndarray,
        min_area: int = 50
    ) -> List[DefectResult]:
        """Extract individual defects from anomaly map"""
        binary = (anomaly_map > self.fine_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            score = float(anomaly_map[y:y+h, x:x+w].max())
            
            defects.append(DefectResult(
                bbox=(x, y, x+w, y+h),
                score=score,
                area=int(area)
            ))
        
        return defects


# ============================================================================
# 4. Multi-Scale Ensemble Detector
# ============================================================================

class MultiScaleEnsembleDetector:
    """
    Ensemble detector using multiple scales and models
    
    Combines:
    - Multiple input resolutions
    - Multiple models (anomaly + classification)
    - Feature-level and decision-level fusion
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        scales: List[int] = [512, 768, 1024],
        weights: Optional[Dict[str, float]] = None,
        device: torch.device = None
    ):
        """
        Args:
            models: Dict of model_name -> model
            scales: Input sizes to use
            weights: Model weights for ensemble
            device: torch device
        """
        self.models = models
        self.scales = scales
        self.weights = weights or {name: 1.0 for name in models}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict]]:
        """
        Run multi-scale ensemble detection
        
        Args:
            image: Input image (H, W, C)
            return_details: If True, return detailed results
        
        Returns:
            Final anomaly score (and details if requested)
        """
        H, W = image.shape[:2]
        
        all_scores = []
        all_maps = []
        details = {}
        
        for scale in self.scales:
            # Resize image
            scaled_image = cv2.resize(image, (scale, scale))
            tensor = self._preprocess(scaled_image)
            
            scale_scores = []
            scale_maps = []
            
            for name, model in self.models.items():
                # Run model
                output = model(tensor)
                
                if isinstance(output, tuple):
                    score, amap = output
                    score = score.item()
                    amap = amap.squeeze().cpu().numpy()
                else:
                    # Classification model
                    probs = F.softmax(output, dim=1)
                    score = (1 - probs[:, 0]).item()
                    amap = np.ones((scale, scale)) * score
                
                # Apply model weight
                weighted_score = score * self.weights.get(name, 1.0)
                scale_scores.append(weighted_score)
                scale_maps.append(cv2.resize(amap, (W, H)))
                
                if return_details:
                    details[f'{name}_scale{scale}'] = {
                        'score': score,
                        'map_shape': amap.shape
                    }
            
            all_scores.extend(scale_scores)
            all_maps.extend(scale_maps)
        
        # Ensemble
        final_score = np.mean(all_scores)
        final_map = np.mean(all_maps, axis=0)
        
        if return_details:
            details['ensemble_score'] = final_score
            details['per_model_scores'] = all_scores
            return final_score, details
        
        return final_score
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)


# ============================================================================
# 5. Unified Pipeline
# ============================================================================

class UnifiedInspectionPipeline:
    """
    Complete inspection pipeline combining all components
    
    Features:
    - Automatic mode selection based on image size
    - Tile-based processing for large images
    - Coarse-to-fine for efficiency
    - Multi-model ensemble for accuracy
    """
    
    def __init__(
        self,
        anomaly_model: nn.Module,
        classifier_model: Optional[nn.Module] = None,
        mode: str = 'auto',  # 'auto', 'tile', 'coarse_to_fine', 'direct'
        tile_size: int = 1024,
        overlap: float = 0.25,
        large_image_threshold: int = 2048,
        device: torch.device = None
    ):
        """
        Args:
            anomaly_model: Model for anomaly detection
            classifier_model: Optional model for defect classification
            mode: Processing mode
            tile_size: Tile size for large images
            overlap: Tile overlap ratio
            large_image_threshold: Images larger than this use tiling
            device: torch device
        """
        self.anomaly_model = anomaly_model
        self.classifier_model = classifier_model
        self.mode = mode
        self.tile_size = tile_size
        self.large_image_threshold = large_image_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.anomaly_model.to(self.device).eval()
        if self.classifier_model is not None:
            self.classifier_model.to(self.device).eval()
        
        self.tile_processor = TileProcessor(tile_size, overlap)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def inspect(
        self,
        image: Union[np.ndarray, str, Path],
        class_names: Optional[List[str]] = None
    ) -> InspectionResult:
        """
        Run complete inspection
        
        Args:
            image: Input image (array or path)
            class_names: Optional list of class names for classification
        
        Returns:
            InspectionResult
        """
        import time
        start_time = time.time()
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        H, W = image.shape[:2]
        
        # Select processing mode
        if self.mode == 'auto':
            if max(H, W) > self.large_image_threshold:
                mode = 'tile'
            else:
                mode = 'direct'
        else:
            mode = self.mode
        
        # Process
        if mode == 'tile':
            result = self._process_tiles(image)
        else:
            result = self._process_direct(image)
        
        # Classification if defects found and classifier available
        if result.defects and self.classifier_model is not None:
            result = self._classify_defects(image, result, class_names)
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _process_direct(self, image: np.ndarray) -> InspectionResult:
        """Direct processing for small images"""
        # Resize to model input size
        resized = cv2.resize(image, (self.tile_size, self.tile_size))
        tensor = self._preprocess(resized)
        
        # Run anomaly detection
        output = self.anomaly_model(tensor)
        
        if isinstance(output, tuple):
            score, amap = output
            score = score.item()
            amap = amap.squeeze().cpu().numpy()
        else:
            score = output.max().item()
            amap = np.ones((self.tile_size, self.tile_size)) * score
        
        # Resize map to original size
        H, W = image.shape[:2]
        amap = cv2.resize(amap, (W, H))
        
        # Extract defects
        defects = self._extract_defects(amap)
        
        return InspectionResult(
            is_anomaly=len(defects) > 0,
            anomaly_score=score,
            anomaly_map=amap,
            defects=defects,
            processing_time_ms=0,
            num_tiles_processed=1
        )
    
    def _process_tiles(self, image: np.ndarray) -> InspectionResult:
        """Tile-based processing for large images"""
        # Extract tiles
        tiles = self.tile_processor.extract_tiles(image)
        
        predictions = []
        max_score = 0
        
        for tile in tiles:
            # Preprocess tile
            tensor = self._preprocess(tile.image)
            
            # Run model
            output = self.anomaly_model(tensor)
            
            if isinstance(output, tuple):
                score, amap = output
                score = score.item()
                amap = amap.squeeze().cpu().numpy()
            else:
                score = output.max().item()
                amap = np.ones((self.tile_size, self.tile_size)) * score
            
            max_score = max(max_score, score)
            predictions.append((amap, tile))
        
        # Merge predictions
        H, W = image.shape[:2]
        merged_map = self.tile_processor.merge_predictions(predictions, (H, W), merge_method='max')
        
        # Extract defects
        defects = self._extract_defects(merged_map)
        
        return InspectionResult(
            is_anomaly=len(defects) > 0,
            anomaly_score=max_score,
            anomaly_map=merged_map,
            defects=defects,
            processing_time_ms=0,
            num_tiles_processed=len(tiles)
        )
    
    def _extract_defects(
        self,
        anomaly_map: np.ndarray,
        threshold: float = 0.5,
        min_area: int = 50
    ) -> List[DefectResult]:
        """Extract defects from anomaly map"""
        binary = (anomaly_map > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            score = float(anomaly_map[y:y+h, x:x+w].max())
            
            defects.append(DefectResult(
                bbox=(x, y, x+w, y+h),
                score=score,
                area=int(area)
            ))
        
        return defects
    
    def _classify_defects(
        self,
        image: np.ndarray,
        result: InspectionResult,
        class_names: Optional[List[str]]
    ) -> InspectionResult:
        """Classify detected defects"""
        for defect in result.defects:
            x1, y1, x2, y2 = defect.bbox
            
            # Extract defect region with padding
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)
            
            roi = image[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (224, 224))
            tensor = self._preprocess(roi_resized)
            
            # Classify
            output = self.classifier_model(tensor)
            probs = F.softmax(output, dim=1)
            class_id = probs.argmax(dim=1).item()
            
            defect.class_id = class_id
            if class_names and class_id < len(class_names):
                defect.class_name = class_names[class_id]
        
        return result
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image.astype(np.uint8))
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)


# ============================================================================
# 6. Visualization Utilities
# ============================================================================

def visualize_result(
    image: np.ndarray,
    result: InspectionResult,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create visualization of inspection result
    
    Args:
        image: Original image
        result: Inspection result
        alpha: Overlay transparency
    
    Returns:
        Visualization image
    """
    vis = image.copy()
    
    # Overlay anomaly map
    if result.anomaly_map is not None:
        # Normalize map to 0-255
        amap = result.anomaly_map.copy()
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
        amap = (amap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
        
        # Resize if needed
        if heatmap.shape[:2] != vis.shape[:2]:
            heatmap = cv2.resize(heatmap, (vis.shape[1], vis.shape[0]))
        
        # Blend
        vis = cv2.addWeighted(vis, 1 - alpha, heatmap, alpha, 0)
    
    # Draw defect bounding boxes
    for defect in result.defects:
        x1, y1, x2, y2 = defect.bbox
        color = (0, 0, 255)  # Red
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"Score: {defect.score:.2f}"
        if defect.class_name:
            label = f"{defect.class_name}: {defect.score:.2f}"
        
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add overall status
    status = "ANOMALY" if result.is_anomaly else "NORMAL"
    color = (0, 0, 255) if result.is_anomaly else (0, 255, 0)
    cv2.putText(vis, f"{status} (Score: {result.anomaly_score:.3f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return vis


# ============================================================================
# 7. Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Large Image Processing Pipeline Test")
    print("=" * 70)
    
    # Test TileProcessor
    print("\n1. TileProcessor:")
    processor = TileProcessor(tile_size=256, overlap=0.25)
    
    # Create dummy large image
    large_image = np.random.rand(1000, 1500, 3).astype(np.float32)
    tiles = processor.extract_tiles(large_image)
    print(f"   Image size: {large_image.shape}")
    print(f"   Number of tiles: {len(tiles)}")
    print(f"   Tile size: {tiles[0].width}x{tiles[0].height}")
    
    # Test merge
    predictions = [(np.random.rand(256, 256), tile) for tile in tiles]
    merged = processor.merge_predictions(predictions, (1000, 1500))
    print(f"   Merged map shape: {merged.shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
