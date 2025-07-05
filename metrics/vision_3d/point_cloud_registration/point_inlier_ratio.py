from typing import Dict, Any
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


class PointInlierRatio(SingleTaskMetric):

    DIRECTIONS = {
        'point_inlier_ratio': +1  # Higher is better
    }

    def __init__(self, use_buffer: bool = True) -> None:
        """Initialize the PointInlierRatio metric.

        Args:
            use_buffer: Whether to use buffer for storing results
        """
        super(PointInlierRatio, self).__init__(use_buffer=use_buffer)

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Override __call__ to handle complex data access pattern.

        This metric uses predicted correspondence points from outputs and correspondences from outputs/labels.
        It can't use the simple SingleTaskMetric pattern because it needs multiple data sources.
        """
        # Extract all required data
        assert 'outputs' in datapoint and 'labels' in datapoint
        outputs = datapoint['outputs']
        labels = datapoint['labels']

        # Validate outputs for predicted correspondence points
        assert 'src_pc' in outputs and 'tgt_pc' in outputs, f"Expected src_pc and tgt_pc in outputs, got {outputs.keys()}"
        src_points = outputs['src_pc']  # Predicted source correspondence points
        tgt_points = outputs['tgt_pc']  # Predicted target correspondence points
        assert isinstance(src_points, torch.Tensor), f"Expected torch.Tensor for src_points, got {type(src_points)}"
        assert isinstance(tgt_points, torch.Tensor), f"Expected torch.Tensor for tgt_points, got {type(tgt_points)}"
        assert src_points.ndim == 2 and src_points.shape[1] == 3, f"{src_points.shape=}"
        assert tgt_points.ndim == 2 and tgt_points.shape[1] == 3, f"{tgt_points.shape=}"

        # Validate outputs - predicted correspondences
        assert 'correspondences' in outputs, f"Expected correspondences in outputs, got {outputs.keys()}"
        pred_correspondences = outputs['correspondences']
        assert isinstance(pred_correspondences, torch.Tensor), f"Expected torch.Tensor for pred_correspondences, got {type(pred_correspondences)}"
        assert (
            pred_correspondences.ndim == 2
            and pred_correspondences.shape[1] == 2
            and pred_correspondences.shape[0] > 0
        ), f"{pred_correspondences.shape=}"

        # Validate labels - ground truth correspondences
        assert 'correspondences' in labels, f"Expected correspondences in labels, got {labels.keys()}"
        gt_correspondences = labels['correspondences']
        assert isinstance(gt_correspondences, torch.Tensor), f"Expected torch.Tensor for gt_correspondences, got {type(gt_correspondences)}"
        assert (
            gt_correspondences.ndim == 2
            and gt_correspondences.shape[1] == 2
            and gt_correspondences.shape[0] > 0
        ), f"{gt_correspondences.shape=}"

        # Compute inlier ratio
        device = src_points.device
        src_length = src_points.shape[0]
        tgt_length = tgt_points.shape[0]

        # Create correspondence map from ground truth
        corr_map = torch.zeros(size=(src_length, tgt_length), device=device)
        corr_map[gt_correspondences[:, 0], gt_correspondences[:, 1]] = 1.0

        # Check predicted correspondences against ground truth map
        point_inlier_ratio = corr_map[pred_correspondences[:, 0], pred_correspondences[:, 1]].mean()

        scores = {'point_inlier_ratio': point_inlier_ratio}
        self.add_to_buffer(scores, datapoint)
        return scores
