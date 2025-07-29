"""
D3Feat Criterion Wrapper for Pylon API Compatibility.

This module provides Pylon-compatible wrapper around the original D3Feat loss functions.
"""

from typing import Dict, Tuple, Any, Optional, List
import torch
import torch.nn as nn

from criteria.base_criterion import BaseCriterion
from criteria.vision_3d.point_cloud_registration.d3feat_criteria.loss import (
    _CircleLoss, _ContrastiveLoss, _DetLoss
)


class D3FeatCriterion(BaseCriterion):
    """Pylon wrapper for D3Feat criterion supporting both CircleLoss and ContrastiveLoss."""
    
    def __init__(
        self,
        loss_type: str = 'circle',
        # Circle loss parameters
        dist_type: str = 'cosine',
        log_scale: float = 10.0,
        safe_radius: float = 0.10,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        pos_optimal: float = 0.1,
        neg_optimal: float = 1.4,
        # Contrastive loss parameters  
        metric: str = 'euclidean',
        # Common parameters
        desc_loss_weight: float = 1.0,
        det_loss_weight: float = 1.0,
        **kwargs
    ):
        """Initialize D3Feat criterion.
        
        Args:
            loss_type: Type of loss to use ('circle' or 'contrastive')
            dist_type: Distance metric type for circle loss
            log_scale: Logarithmic scale factor for circle loss
            safe_radius: Safe radius for correspondences
            pos_margin: Positive margin
            neg_margin: Negative margin
            pos_optimal: Positive optimal value for circle loss
            neg_optimal: Negative optimal value for circle loss
            metric: Distance metric for contrastive loss
            desc_loss_weight: Descriptor loss weight
            det_loss_weight: Detection loss weight
        """
        super(D3FeatCriterion, self).__init__(**kwargs)
        
        assert loss_type in ['circle', 'contrastive'], f"loss_type must be 'circle' or 'contrastive', got {loss_type}"
        
        self.loss_type = loss_type
        self.desc_loss_weight = desc_loss_weight
        self.det_loss_weight = det_loss_weight
        
        # Set DIRECTIONS based on loss type
        if loss_type == 'circle':
            self.DIRECTIONS = {"circle_loss": -1}  # Lower is better
        else:
            self.DIRECTIONS = {"contrastive_loss": -1}  # Lower is better
        
        # Initialize loss functions based on type
        if loss_type == 'circle':
            self.descriptor_loss = _CircleLoss(
                dist_type=dist_type,
                log_scale=log_scale,
                safe_radius=safe_radius,
                pos_margin=pos_margin,
                neg_margin=neg_margin
            )
            # Override optimal values if provided
            self.descriptor_loss.pos_optimal = pos_optimal
            self.descriptor_loss.neg_optimal = neg_optimal
            self.det_loss = _DetLoss()
        else:  # contrastive
            self.descriptor_loss = _ContrastiveLoss(
                pos_margin=pos_margin,
                neg_margin=neg_margin,
                metric=metric,
                safe_radius=safe_radius
            )
            self.det_loss = _DetLoss(metric=metric)
        
    def __call__(
        self, 
        y_pred: Dict[str, torch.Tensor], 
        y_true: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute D3Feat losses and add to buffer.
        
        Args:
            y_pred: Model predictions with 'descriptors' and 'scores'
            y_true: Ground truth with 'correspondences' and potentially other keys
            
        Returns:
            Single loss tensor
        """
        # Extract predictions
        descriptors = y_pred['descriptors']  # [N_total, feature_dim]
        scores = y_pred['scores']            # [N_total, 1]
        
        # Extract ground truth
        correspondences = y_true['correspondences']  # [K, 2]
        
        # Get batch lengths (assuming batch size = 1 for now)
        # In actual implementation, this should come from batch metadata
        N_total = descriptors.shape[0]
        N_src = N_total // 2  # Assuming equal src/tgt sizes
        N_tgt = N_total - N_src
        
        # Split descriptors and scores
        desc_src = descriptors[:N_src]     # [N_src, feature_dim]
        desc_tgt = descriptors[N_src:]     # [N_tgt, feature_dim]
        scores_src = scores[:N_src]        # [N_src, 1]
        scores_tgt = scores[N_src:]        # [N_tgt, 1]
        
        # Get corresponding descriptors based on correspondences
        if correspondences.numel() > 0:
            corr_src_idx = correspondences[:, 0].long()
            corr_tgt_idx = correspondences[:, 1].long()
            
            anchor_desc = desc_src[corr_src_idx]      # [K, feature_dim]
            positive_desc = desc_tgt[corr_tgt_idx]    # [K, feature_dim]
            anchor_scores = scores_src[corr_src_idx]  # [K, 1]
            positive_scores = scores_tgt[corr_tgt_idx] # [K, 1]
            
            # Use the provided dist_keypts from ground truth (like original D3Feat)
            # Note: In actual training, dist_keypts should come from y_true, but for now
            # we'll create a placeholder. This should be fixed when integrating with proper data pipeline.
            if 'dist_keypts' in y_true:
                dist_keypts = y_true['dist_keypts']
            else:
                # Fallback: create identity matrix as in original D3Feat when no distance provided
                num_corr = anchor_desc.shape[0]
                dist_keypts = torch.eye(num_corr, device=anchor_desc.device)
            
            # Compute descriptor loss
            desc_loss, accuracy, furthest_pos, avg_neg, _, dists = self.descriptor_loss(
                anchor_desc, positive_desc, dist_keypts
            )
            
            # Compute detection loss
            det_loss = self.det_loss(dists, anchor_scores, positive_scores)
            
            # Combined loss
            total_loss = (self.desc_loss_weight * desc_loss + 
                         self.det_loss_weight * det_loss)
        else:
            # No correspondences available
            total_loss = torch.tensor(0.0, device=descriptors.device, requires_grad=True)
            desc_loss = torch.tensor(0.0, device=descriptors.device)
            det_loss = torch.tensor(0.0, device=descriptors.device)
            accuracy = 0.0
        
        # Add to buffer if enabled
        if self.use_buffer:
            self.add_to_buffer(total_loss)
        return total_loss
    
    def summarize(self, output_path: Optional[str] = None) -> Dict[str, float]:
        """Summarize losses from buffer.
        
        Returns:
            Dictionary with loss statistics
        """
        assert self.use_buffer, "Buffer must be enabled to summarize losses"
            
        # Wait for buffer to be processed
        self._buffer_queue.join()
        
        with self._buffer_lock:
            assert len(self.buffer) > 0, "Buffer is empty - no losses to summarize"
            
            losses = torch.stack(self.buffer)
            avg_loss = float(losses.mean())
            
        if self.loss_type == 'circle':
            return {"circle_loss": avg_loss}
        else:
            return {"contrastive_loss": avg_loss}