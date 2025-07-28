"""
D3Feat Criterion Wrapper for Pylon API Compatibility.

This module provides Pylon-compatible wrappers around the original D3Feat loss functions.
"""

from typing import Dict, Tuple, Any
import torch
import torch.nn as nn

from criteria.base_criterion import BaseCriterion
from criteria.vision_3d.point_cloud_registration.d3feat_criteria.loss import (
    _CircleLoss, _ContrastiveLoss, _DetLoss
)


class CircleLoss(BaseCriterion):
    """Pylon wrapper for D3Feat CircleLoss criterion."""
    
    DIRECTIONS = {"circle_loss": -1}  # Lower is better
    
    def __init__(
        self,
        dist_type: str = 'cosine',
        log_scale: float = 10.0,
        safe_radius: float = 0.10,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        desc_loss_weight: float = 1.0,
        det_loss_weight: float = 1.0,
        **kwargs
    ):
        """Initialize CircleLoss criterion.
        
        Args:
            dist_type: Distance metric type
            log_scale: Logarithmic scale factor
            safe_radius: Safe radius for correspondences
            pos_margin: Positive margin
            neg_margin: Negative margin
            desc_loss_weight: Descriptor loss weight
            det_loss_weight: Detection loss weight
        """
        super(CircleLoss, self).__init__(**kwargs)
        
        self.desc_loss_weight = desc_loss_weight
        self.det_loss_weight = det_loss_weight
        
        # Initialize original D3Feat losses
        self.circle_loss = _CircleLoss(
            dist_type=dist_type,
            log_scale=log_scale,
            safe_radius=safe_radius,
            pos_margin=pos_margin,
            neg_margin=neg_margin
        )
        self.det_loss = _DetLoss()
        
    def _compute_loss(
        self, 
        y_pred: Dict[str, torch.Tensor], 
        y_true: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute D3Feat losses.
        
        Args:
            y_pred: Model predictions with 'descriptors' and 'scores'
            y_true: Ground truth with 'correspondences' and potentially other keys
            
        Returns:
            Dictionary with loss components
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
            
            # Compute keypoint distances (dummy for now)
            # In practice, this should come from the dataset
            dist_keypts = torch.ones((anchor_desc.shape[0], anchor_desc.shape[0]), 
                                   device=anchor_desc.device) * 0.5
            
            # Compute descriptor loss
            desc_loss, accuracy, furthest_pos, avg_neg, _, dists = self.circle_loss(
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
        
        return {
            'circle_loss': total_loss,
            'desc_loss': desc_loss,
            'det_loss': det_loss,
            'accuracy': torch.tensor(accuracy, device=descriptors.device)
        }


class ContrastiveLoss(BaseCriterion):
    """Pylon wrapper for D3Feat ContrastiveLoss criterion."""
    
    DIRECTIONS = {"contrastive_loss": -1}  # Lower is better
    
    def __init__(
        self,
        pos_margin: float = 0.1,
        neg_margin: float = 1.4,
        metric: str = 'euclidean',
        safe_radius: float = 0.25,
        desc_loss_weight: float = 1.0,
        det_loss_weight: float = 1.0,
        **kwargs
    ):
        """Initialize ContrastiveLoss criterion.
        
        Args:
            pos_margin: Positive margin
            neg_margin: Negative margin
            metric: Distance metric
            safe_radius: Safe radius for correspondences
            desc_loss_weight: Descriptor loss weight
            det_loss_weight: Detection loss weight
        """
        super(ContrastiveLoss, self).__init__(**kwargs)
        
        self.desc_loss_weight = desc_loss_weight
        self.det_loss_weight = det_loss_weight
        
        # Initialize original D3Feat losses
        self.contrastive_loss = _ContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            metric=metric,
            safe_radius=safe_radius
        )
        self.det_loss = _DetLoss(metric=metric)
        
    def _compute_loss(
        self, 
        y_pred: Dict[str, torch.Tensor], 
        y_true: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute D3Feat contrastive losses.
        
        Args:
            y_pred: Model predictions with 'descriptors' and 'scores'
            y_true: Ground truth with 'correspondences'
            
        Returns:
            Dictionary with loss components
        """
        # Extract predictions
        descriptors = y_pred['descriptors']  # [N_total, feature_dim]
        scores = y_pred['scores']            # [N_total, 1]
        
        # Extract ground truth
        correspondences = y_true['correspondences']  # [K, 2]
        
        # Get batch lengths (assuming batch size = 1 for now)
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
            
            # Compute keypoint distances (dummy for now)
            dist_keypts = torch.ones((anchor_desc.shape[0], anchor_desc.shape[0]), 
                                   device=anchor_desc.device) * 0.5
            
            # Compute descriptor loss
            desc_loss, accuracy, furthest_pos, avg_neg, _, dists = self.contrastive_loss(
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
        
        return {
            'contrastive_loss': total_loss,
            'desc_loss': desc_loss,
            'det_loss': det_loss,
            'accuracy': torch.tensor(accuracy, device=descriptors.device)
        }


# Default D3Feat criterion (using CircleLoss)
D3FeatCriterion = CircleLoss