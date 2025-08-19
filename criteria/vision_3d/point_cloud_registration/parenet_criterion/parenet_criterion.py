"""
PARENet Criterion Wrapper for Pylon API Compatibility.

This module provides Pylon-compatible wrapper around the original PARENet loss functions.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from easydict import EasyDict

from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import _OverallLoss


class PARENetCriterion(SingleTaskCriterion):
    """Pylon wrapper for PARENet criterion supporting multi-component loss."""

    def __init__(
        self,
        # Coarse loss parameters
        coarse_positive_margin: float = 0.1,
        coarse_negative_margin: float = 1.4,
        coarse_positive_optimal: float = 0.1,
        coarse_negative_optimal: float = 1.4,
        coarse_log_scale: float = 10.0,
        coarse_positive_overlap: float = 0.1,

        # Fine loss parameters
        fine_positive_radius: float = 0.1,
        fine_negative_radius: float = 0.1,
        fine_positive_margin: float = 0.05,
        fine_negative_margin: float = 0.2,

        # Loss weights
        weight_coarse_loss: float = 1.0,
        weight_fine_ri_loss: float = 1.0,
        weight_fine_re_loss: float = 1.0,

        **kwargs
    ):
        """Initialize PARENet criterion.

        Args:
            coarse_positive_margin: Positive margin for coarse loss
            coarse_negative_margin: Negative margin for coarse loss
            coarse_positive_optimal: Positive optimal value for coarse loss
            coarse_negative_optimal: Negative optimal value for coarse loss
            coarse_log_scale: Log scale for coarse loss
            coarse_positive_overlap: Positive overlap threshold for coarse loss
            fine_positive_radius: Positive radius for fine loss
            fine_negative_radius: Negative radius for fine loss
            fine_positive_margin: Positive margin for fine loss
            fine_negative_margin: Negative margin for fine loss
            weight_coarse_loss: Weight for coarse loss
            weight_fine_ri_loss: Weight for fine rotation-invariant loss
            weight_fine_re_loss: Weight for fine rotation-equivariant loss
        """
        super(PARENetCriterion, self).__init__(**kwargs)

        # All loss components are minimized (lower is better)
        self.DIRECTIONS = {
            "loss": -1,        # Total loss - lower is better
            "c_loss": -1,      # Coarse loss - lower is better
            "f_ri_loss": -1,   # Fine RI loss - lower is better
            "f_re_loss": -1    # Fine RE loss - lower is better
        }

        # Build PARENet configuration using EasyDict
        cfg = EasyDict()

        # Coarse loss configuration
        cfg.coarse_loss = EasyDict()
        cfg.coarse_loss.positive_margin = coarse_positive_margin
        cfg.coarse_loss.negative_margin = coarse_negative_margin
        cfg.coarse_loss.positive_optimal = coarse_positive_optimal
        cfg.coarse_loss.negative_optimal = coarse_negative_optimal
        cfg.coarse_loss.log_scale = coarse_log_scale
        cfg.coarse_loss.positive_overlap = coarse_positive_overlap

        # Fine loss configuration
        cfg.fine_loss = EasyDict()
        cfg.fine_loss.positive_radius = fine_positive_radius
        cfg.fine_loss.negative_radius = fine_negative_radius
        cfg.fine_loss.positive_margin = fine_positive_margin
        cfg.fine_loss.negative_margin = fine_negative_margin

        # Loss weights configuration
        cfg.loss = EasyDict()
        cfg.loss.weight_coarse_loss = weight_coarse_loss
        cfg.loss.weight_fine_ri_loss = weight_fine_ri_loss
        cfg.loss.weight_fine_re_loss = weight_fine_re_loss

        # Initialize the original PARENet loss
        self.parenet_loss = _OverallLoss(cfg)
        self.cfg = cfg

    def __call__(
        self,
        y_pred: Dict[str, torch.Tensor],
        y_true: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute PARENet losses and add to buffer.

        Args:
            y_pred: Model predictions from PARENetModel containing:
                - ref_feats_c: Reference coarse features
                - src_feats_c: Source coarse features
                - gt_node_corr_indices: Ground truth node correspondences
                - gt_node_corr_overlaps: Ground truth node overlaps
                - matching_scores: Fine matching scores
                - ref_node_corr_knn_points: Reference node correspondence points
                - src_node_corr_knn_points: Source node correspondence points
                - ref_node_corr_knn_masks: Reference node correspondence masks
                - src_node_corr_knn_masks: Source node correspondence masks
                - re_ref_node_corr_knn_feats: RE reference features
                - re_src_node_corr_knn_feats: RE source features
            y_true: Ground truth with:
                - transform: Ground truth transformation matrix [4, 4]

        Returns:
            Single loss tensor (weighted combination of coarse + fine RI + fine RE)
        """
        # Validate inputs
        assert isinstance(y_pred, dict), f"Expected dict for y_pred, got {type(y_pred)}"
        assert isinstance(y_true, dict), f"Expected dict for y_true, got {type(y_true)}"

        # Check required keys in y_pred
        required_pred_keys = [
            'ref_feats_c', 'src_feats_c', 'gt_node_corr_indices', 'gt_node_corr_overlaps',
            'matching_scores', 'ref_node_corr_knn_points', 'src_node_corr_knn_points',
            'ref_node_corr_knn_masks', 'src_node_corr_knn_masks',
            'ref_node_corr_knn_scores', 'src_node_corr_knn_scores',  # Required by loss
            're_ref_node_corr_knn_feats', 're_src_node_corr_knn_feats'
        ]
        for key in required_pred_keys:
            assert key in y_pred, f"Missing required key '{key}' in y_pred. Available keys: {list(y_pred.keys())}"

        # Check required keys in y_true
        assert 'transform' in y_true, f"Missing required key 'transform' in y_true. Available keys: {list(y_true.keys())}"

        # Prepare data structures for original PARENet loss
        # The original loss expects output_dict and data_dict
        output_dict = {
            'ref_feats_c': y_pred['ref_feats_c'],
            'src_feats_c': y_pred['src_feats_c'],
            'gt_node_corr_indices': y_pred['gt_node_corr_indices'],
            'gt_node_corr_overlaps': y_pred['gt_node_corr_overlaps'],
            'matching_scores': y_pred['matching_scores'],
            'ref_node_corr_knn_points': y_pred['ref_node_corr_knn_points'],
            'src_node_corr_knn_points': y_pred['src_node_corr_knn_points'],
            'ref_node_corr_knn_masks': y_pred['ref_node_corr_knn_masks'],
            'src_node_corr_knn_masks': y_pred['src_node_corr_knn_masks'],
            'ref_node_corr_knn_scores': y_pred['ref_node_corr_knn_scores'],  # Required by loss
            'src_node_corr_knn_scores': y_pred['src_node_corr_knn_scores'],  # Required by loss
            're_ref_node_corr_knn_feats': y_pred['re_ref_node_corr_knn_feats'],
            're_src_node_corr_knn_feats': y_pred['re_src_node_corr_knn_feats'],
        }

        data_dict = {
            'transform': y_true['transform']
        }

        # Compute loss using original PARENet loss
        loss_dict = self.parenet_loss(output_dict, data_dict)

        # Extract the total loss
        total_loss = loss_dict['loss']

        # Add to buffer if enabled
        self.add_to_buffer(total_loss)

        return total_loss
