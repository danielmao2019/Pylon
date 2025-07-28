"""
D3Feat Metrics for Pylon Integration.

This module provides Pylon-compatible wrappers around D3Feat evaluation metrics.
"""

from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn

from metrics.wrappers.single_task_metric import SingleTaskMetric
from metrics.base_metric import BaseMetric
from metrics.vision_3d.point_cloud_registration.d3feat_metrics.metrics import (
    calculate_acc, calculate_iou, calculate_iou_single_shape, IoU_from_confusions
)


class D3FeatAccuracyMetric(SingleTaskMetric):
    """D3Feat accuracy metric for descriptor matching."""
    
    DIRECTIONS = {"accuracy": +1}  # Higher is better
    
    def __init__(self, use_buffer: bool = True, **kwargs):
        """Initialize D3Feat accuracy metric.
        
        Args:
            use_buffer: Whether to use buffer for storing results
        """
        super(D3FeatAccuracyMetric, self).__init__(use_buffer=use_buffer, **kwargs)
        
    def _compute_score(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute accuracy score.
        
        Args:
            y_pred: Predicted logits [N, num_classes]
            y_true: Ground truth labels [N]
            
        Returns:
            Dictionary with accuracy score
        """
        accuracy = calculate_acc(y_pred, y_true)
        
        return {
            'accuracy': torch.scalar_tensor(accuracy, dtype=torch.float32, device=y_pred.device)
        }


class D3FeatIoUMetric(SingleTaskMetric):
    """D3Feat IoU metric for segmentation tasks."""
    
    DIRECTIONS = {"iou": +1}  # Higher is better
    
    def __init__(self, num_classes: int, use_buffer: bool = True, **kwargs):
        """Initialize D3Feat IoU metric.
        
        Args:
            num_classes: Number of classes for IoU computation
            use_buffer: Whether to use buffer for storing results
        """
        super(D3FeatIoUMetric, self).__init__(use_buffer=use_buffer, **kwargs)
        self.num_classes = num_classes
        
    def _compute_score(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor,
        stack_lengths: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Compute IoU score.
        
        Args:
            y_pred: Predicted logits [N, num_classes]
            y_true: Ground truth labels [N]
            stack_lengths: Optional batch lengths for batched computation
            
        Returns:
            Dictionary with IoU scores
        """
        if stack_lengths is not None:
            # Batched computation
            iou_scores = calculate_iou(
                y_pred, y_true, 
                stack_lengths.detach().cpu().numpy(), 
                self.num_classes
            )
            iou_tensor = torch.from_numpy(iou_scores).to(dtype=torch.float32, device=y_pred.device)
            
            # Return mean IoU and per-class IoU
            result = {'iou': iou_tensor.mean()}
            for i, score in enumerate(iou_tensor):
                result[f'iou_class_{i}'] = score
                
        else:
            # Single shape computation
            iou_scores = calculate_iou_single_shape(y_pred, y_true, self.num_classes)
            iou_tensor = torch.from_numpy(iou_scores).to(dtype=torch.float32, device=y_pred.device)
            
            result = {'iou': iou_tensor.mean()}
            for i, score in enumerate(iou_tensor):
                result[f'iou_class_{i}'] = score
                
        return result


class D3FeatDescriptorMetric(BaseMetric):
    """D3Feat descriptor evaluation metric for feature matching."""
    
    DIRECTIONS = {
        "desc_matching_accuracy": +1,    # Higher is better
        "feature_match_recall": +1,      # Higher is better
        "desc_distance": -1,             # Lower is better
    }
    
    def __init__(
        self, 
        distance_threshold: float = 0.1,
        use_buffer: bool = True, 
        **kwargs
    ):
        """Initialize D3Feat descriptor metric.
        
        Args:
            distance_threshold: Distance threshold for positive matches
            use_buffer: Whether to use buffer for storing results
        """
        super(D3FeatDescriptorMetric, self).__init__(use_buffer=use_buffer, **kwargs)
        self.distance_threshold = distance_threshold

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Compute D3Feat descriptor metrics.
        
        Args:
            datapoint: Dictionary containing:
                - outputs: {'descriptors': tensor, 'scores': tensor}  
                - labels: {'correspondences': tensor}
                - meta_info: metadata
                
        Returns:
            Dictionary with descriptor matching scores
        """
        # Extract outputs and labels from datapoint
        assert 'outputs' in datapoint and 'labels' in datapoint
        y_pred = datapoint['outputs']
        y_true = datapoint['labels']
        
        scores = self._compute_score(y_pred, y_true)
        self.add_to_buffer(scores, datapoint)
        return scores
        
    def _compute_score(
        self, 
        y_pred: Dict[str, torch.Tensor], 
        y_true: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute descriptor matching metrics.
        
        Args:
            y_pred: Dictionary with 'descriptors' and 'scores'
            y_true: Dictionary with 'correspondences'
            
        Returns:
            Dictionary with descriptor metrics
        """
        descriptors = y_pred['descriptors']  # [N, feature_dim]
        correspondences = y_true['correspondences']  # [K, 2]
        
        if correspondences.numel() == 0:
            # No correspondences to evaluate
            device = descriptors.device
            return {
                'desc_matching_accuracy': torch.scalar_tensor(0.0, device=device),
                'feature_match_recall': torch.scalar_tensor(0.0, device=device),
                'desc_distance': torch.scalar_tensor(float('inf'), device=device),
            }
        
        # Assume descriptors are [src_descriptors; tgt_descriptors]
        N_total = descriptors.shape[0]
        N_src = N_total // 2
        
        src_desc = descriptors[:N_src]  # [N_src, feature_dim]
        tgt_desc = descriptors[N_src:]  # [N_tgt, feature_dim]
        
        # Get corresponding descriptor pairs
        corr_src_idx = correspondences[:, 0].long()
        corr_tgt_idx = correspondences[:, 1].long()
        
        # Ensure indices are valid
        valid_src = (corr_src_idx >= 0) & (corr_src_idx < N_src)
        valid_tgt = (corr_tgt_idx >= 0) & (corr_tgt_idx < (N_total - N_src))
        valid_mask = valid_src & valid_tgt
        
        if valid_mask.sum() == 0:
            device = descriptors.device
            return {
                'desc_matching_accuracy': torch.scalar_tensor(0.0, device=device),
                'feature_match_recall': torch.scalar_tensor(0.0, device=device),
                'desc_distance': torch.scalar_tensor(float('inf'), device=device),
            }
        
        # Filter valid correspondences
        corr_src_idx = corr_src_idx[valid_mask]
        corr_tgt_idx = corr_tgt_idx[valid_mask]
        
        # Get corresponding descriptors
        corr_src_desc = src_desc[corr_src_idx]  # [K_valid, feature_dim]
        corr_tgt_desc = tgt_desc[corr_tgt_idx]  # [K_valid, feature_dim]
        
        # Compute distances between corresponding descriptors
        desc_distances = torch.norm(corr_src_desc - corr_tgt_desc, p=2, dim=1)  # [K_valid]
        
        # Compute metrics
        mean_distance = desc_distances.mean()
        matching_accuracy = (desc_distances < self.distance_threshold).float().mean()
        
        # Feature match recall (how many correspondences have close descriptors)
        feature_match_recall = matching_accuracy  # Same as accuracy for this metric
        
        return {
            'desc_matching_accuracy': matching_accuracy,
            'feature_match_recall': feature_match_recall,
            'desc_distance': mean_distance,
        }

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Summarize descriptor metrics across all data points."""
        assert self.use_buffer and hasattr(self, 'buffer') and self.buffer is not None
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

        from utils.ops.dict_as_tensor import transpose_buffer
        from utils.input_checks.str_types import check_write_file
        from utils.io.json import save_json

        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
        # summarize scores
        result: Dict[str, Dict[str, torch.Tensor]] = {
            "aggregated": {},
            "per_datapoint": {},
        }

        # For each metric, store both the per-datapoint values and compute the mean
        for key in buffer:
            key_scores = torch.stack(buffer[key], dim=0)
            assert key_scores.ndim == 1, f"{key=}, {key_scores.shape=}"
            # Store per-datapoint values
            result["per_datapoint"][key] = key_scores
            # Store aggregated value
            result["aggregated"][key] = key_scores.mean()

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result


# Default D3Feat metric (descriptor metric)
D3FeatMetric = D3FeatDescriptorMetric