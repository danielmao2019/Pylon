from typing import Dict, Any
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.point_cloud_ops.apply_transform import apply_transform


class InlierRatio(SingleTaskMetric):
    """
    Inlier Ratio metric for 3D point cloud registration.

    Inlier Ratio measures the proportion of points in the source point cloud
    that are within a certain distance threshold of their nearest neighbors in
    the target point cloud after applying the predicted transformation.
    """

    DIRECTION = +1  # Higher is better

    def __init__(self, threshold: float, use_buffer: bool = True) -> None:
        """
        Initialize the Inlier Ratio metric.

        Args:
            threshold: Distance threshold for considering a point as an inlier
            use_buffer: Whether to use buffer for storing results
        """
        super(InlierRatio, self).__init__(use_buffer=use_buffer)
        self.threshold = threshold

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Compute inlier ratio from a complete datapoint.

        For correspondence-based PCR algorithms that predict point correspondences,
        this metric evaluates correspondence quality using the ground truth transform.

        Args:
            datapoint: Complete datapoint containing:
                - 'outputs': {'src_pc': predicted_source_points, 'tgt_pc': predicted_target_correspondences}
                - 'labels': {'transform': ground_truth_transform_matrix}
                - 'meta_info': {...}

        Returns:
            Dictionary containing inlier ratio score
        """
        # Extract predicted correspondences from outputs
        assert 'outputs' in datapoint, "Missing 'outputs' in datapoint"
        outputs = datapoint['outputs']
        assert 'src_pc' in outputs and 'tgt_pc' in outputs, f"Missing point clouds in outputs: {outputs.keys()}"

        src_points = outputs['src_pc']  # Predicted source points
        tgt_points = outputs['tgt_pc']  # Predicted target correspondences

        # Handle different input formats - could be tensors or dicts with 'pos'
        if isinstance(src_points, dict):
            assert 'pos' in src_points, "Missing 'pos' in src_pc"
            src_points = src_points['pos']
        if isinstance(tgt_points, dict):
            assert 'pos' in tgt_points, "Missing 'pos' in tgt_pc"
            tgt_points = tgt_points['pos']

        # Handle batched input
        if src_points.ndim == 3:
            assert src_points.shape[0] == 1, f"Expected batch size 1, got {src_points.shape[0]}"
            src_points = src_points.squeeze(0)  # (N, 3)
        if tgt_points.ndim == 3:
            assert tgt_points.shape[0] == 1, f"Expected batch size 1, got {tgt_points.shape[0]}"
            tgt_points = tgt_points.squeeze(0)  # (N, 3)

        assert src_points.ndim == 2 and src_points.shape[1] == 3, f"Expected (N, 3) source points, got {src_points.shape}"
        assert tgt_points.ndim == 2 and tgt_points.shape[1] == 3, f"Expected (N, 3) target points, got {tgt_points.shape}"
        assert src_points.shape[0] == tgt_points.shape[0], f"Mismatched correspondence count: {src_points.shape[0]} vs {tgt_points.shape[0]}"

        # Extract ground truth transformation from labels
        assert 'labels' in datapoint, "Missing 'labels' in datapoint"
        labels = datapoint['labels']
        assert 'transform' in labels, f"Missing transform in labels: {labels.keys()}"
        gt_transform = labels['transform']  # (1, 4, 4)

        # Handle batch dimension
        if gt_transform.ndim == 3:
            assert gt_transform.shape[0] == 1, f"Expected batch size 1, got {gt_transform.shape[0]}"
            gt_transform = gt_transform.squeeze(0)  # (4, 4)
        assert gt_transform.shape == (4, 4), f"Expected (4, 4) transform, got {gt_transform.shape}"

        # Apply ground truth transformation to predicted source points
        transformed_src_points = apply_transform(src_points, gt_transform)

        # Compute distances between transformed source points and their predicted target correspondences
        # Since these are correspondences (not arbitrary target points), we compute point-to-point distances
        distances = torch.norm(transformed_src_points - tgt_points, dim=1)  # (N,)

        # Compute inlier mask and ratio
        inlier_mask = (distances <= self.threshold)
        inlier_ratio = torch.mean(inlier_mask.float())

        scores = {'inlier_ratio': inlier_ratio}
        self.add_to_buffer(scores, datapoint)
        return scores
