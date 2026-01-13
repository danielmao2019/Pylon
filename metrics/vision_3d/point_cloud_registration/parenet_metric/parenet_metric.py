"""
PARENet Metric Wrapper for Pylon API Compatibility.

This module provides Pylon-compatible wrapper around PARENet evaluation metrics.
"""

from typing import Dict, Any, Optional
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError
from metrics.vision_3d.point_cloud_registration.inlier_ratio import InlierRatio
from criteria.vision_3d.point_cloud_registration.parenet_criterion.loss import _Evaluator
from easydict import EasyDict


class PARENetMetric(SingleTaskMetric):
    """Pylon wrapper for PARENet evaluation metrics combining multiple metrics."""

    # Define DIRECTIONS for all metrics (instance-level as this is a wrapper)
    def __init__(
        self,
        # Inlier ratio parameters
        inlier_threshold: float = 0.1,

        # Evaluation parameters (for PARENet-specific metrics)
        acceptance_overlap: float = 0.1,
        acceptance_radius: float = 0.1,
        rmse_threshold: float = 0.2,
        feat_rre_threshold: float = 30.0,

        **kwargs
    ):
        """Initialize PARENet metric wrapper.

        Args:
            inlier_threshold: Distance threshold for inlier ratio computation
            acceptance_overlap: Acceptance overlap threshold for coarse matching
            acceptance_radius: Acceptance radius for fine matching
            rmse_threshold: RMSE threshold for registration recall
            feat_rre_threshold: Feature RRE threshold
        """
        super(PARENetMetric, self).__init__(**kwargs)

        # Define DIRECTIONS for all metrics we compute
        self.DIRECTIONS = {
            "rotation_error": -1,      # RRE - lower is better
            "translation_error": -1,   # RTE - lower is better
            "inlier_ratio": 1,         # IR - higher is better
            "point_inlier_ratio": 1,   # PIR - higher is better (coarse precision)
            "fine_precision": 1,       # Fine precision - higher is better
            "rmse": -1,                # RMSE - lower is better
            "registration_recall": 1    # RR - higher is better
        }

        # Initialize existing Pylon metrics
        self.isotropic_error = IsotropicTransformError(use_buffer=False)
        self.inlier_ratio = InlierRatio(threshold=inlier_threshold, use_buffer=False)

        # Build PARENet configuration for evaluator
        cfg = EasyDict()
        cfg.eval = EasyDict()
        cfg.eval.acceptance_overlap = acceptance_overlap
        cfg.eval.acceptance_radius = acceptance_radius
        cfg.eval.rmse_threshold = rmse_threshold
        cfg.eval.feat_rre_threshold = feat_rre_threshold

        # Initialize PARENet evaluator for additional metrics
        self.parenet_evaluator = _Evaluator(cfg)

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Compute all PARENet metrics from a complete datapoint.

        Args:
            datapoint: Complete datapoint containing:
                - 'outputs': Model outputs from PARENetModel
                - 'labels': Ground truth labels
                - 'meta_info': Metadata including 'idx'

        Returns:
            Dictionary containing all metric scores
        """
        # Validate inputs
        assert 'outputs' in datapoint, "Missing 'outputs' in datapoint"
        assert 'labels' in datapoint, "Missing 'labels' in datapoint"

        outputs = datapoint['outputs']
        labels = datapoint['labels']

        # Compute transform error using existing Pylon metric
        # Create single-key dictionaries for SingleTaskMetric compatibility
        transform_outputs = {'estimated_transform': outputs['estimated_transform']}
        transform_labels = {'transform': labels['transform']}

        # The IsotropicTransformError metric expects single tensors, so we extract them
        estimated_transform = outputs['estimated_transform']
        gt_transform = labels['transform']

        # Handle batch dimension if present
        if estimated_transform.ndim == 3:
            assert estimated_transform.shape[0] == 1, f"Expected batch size 1, got {estimated_transform.shape[0]}"
            estimated_transform = estimated_transform.squeeze(0)
        if gt_transform.ndim == 3:
            assert gt_transform.shape[0] == 1, f"Expected batch size 1, got {gt_transform.shape[0]}"
            gt_transform = gt_transform.squeeze(0)

        # Compute isotropic transform errors
        transform_errors = self.isotropic_error._compute_score(estimated_transform, gt_transform)

        # Compute inlier ratio using existing Pylon metric
        # Create a temporary datapoint for the inlier ratio metric
        inlier_datapoint = {
            'outputs': {
                'src_pc': outputs['ref_corr_points'],  # Reference points (will be transformed)
                'tgt_pc': outputs['src_corr_points']   # Source points (targets after transformation)
            },
            'labels': {
                'transform': labels['transform']
            },
            'meta_info': datapoint['meta_info']
        }
        inlier_scores = self.inlier_ratio(inlier_datapoint)

        # Prepare data for PARENet evaluator (requires output_dict and data_dict format)
        # Pass outputs directly - evaluator can handle extra keys
        data_dict = {
            'transform': labels['transform']
        }

        # Compute additional PARENet metrics
        parenet_scores = self.parenet_evaluator(outputs, data_dict)

        # Combine all scores
        scores = {
            'rotation_error': transform_errors['rotation_error'],      # RRE
            'translation_error': transform_errors['translation_error'], # RTE
            'inlier_ratio': inlier_scores['inlier_ratio'],             # IR
            'point_inlier_ratio': parenet_scores['PIR'],               # PIR (coarse precision)
            'fine_precision': parenet_scores['IR'],                    # Fine precision
            'rmse': parenet_scores['RMSE'],                            # RMSE
            'registration_recall': parenet_scores['RR']                # Registration recall
        }

        # Add to buffer
        self.add_to_buffer(scores, datapoint)

        return scores
