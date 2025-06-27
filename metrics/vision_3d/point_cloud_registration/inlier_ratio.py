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

        Args:
            datapoint: Complete datapoint containing:
                - 'inputs': {'src_pc': {'pos': ...}, 'tgt_pc': {'pos': ...}}
                - 'outputs': {'transform': predicted_transform_matrix}
                - 'labels': {'transform': ground_truth_transform_matrix}
                - 'meta_info': {...}

        Returns:
            Dictionary containing inlier ratio score
        """
        # Extract source and target point clouds from inputs
        assert 'inputs' in datapoint, "Missing 'inputs' in datapoint"
        inputs = datapoint['inputs']
        assert 'src_pc' in inputs and 'tgt_pc' in inputs, f"Missing point clouds in inputs: {inputs.keys()}"
        assert 'pos' in inputs['src_pc'] and 'pos' in inputs['tgt_pc'], "Missing 'pos' in point clouds"

        src_points = inputs['src_pc']['pos']  # (B, N, 3)
        tgt_points = inputs['tgt_pc']['pos']  # (B, M, 3)

        # Handle batched input
        assert src_points.ndim == 3 and src_points.shape[0] == 1, f"Expected batched input with B=1, got {src_points.shape}"
        assert tgt_points.ndim == 3 and tgt_points.shape[0] == 1, f"Expected batched input with B=1, got {tgt_points.shape}"
        assert src_points.shape[2] == 3, f"Expected 3D points, got {src_points.shape}"
        assert tgt_points.shape[2] == 3, f"Expected 3D points, got {tgt_points.shape}"

        # Remove batch dimension
        src_points = src_points.squeeze(0)  # (N, 3)
        tgt_points = tgt_points.squeeze(0)  # (M, 3)

        # Extract predicted transformation from outputs
        assert 'outputs' in datapoint, "Missing 'outputs' in datapoint"
        outputs = datapoint['outputs']
        assert 'transform' in outputs, f"Missing transform in outputs: {outputs.keys()}"
        predicted_transform = outputs['transform']  # (1, 4, 4)
        assert predicted_transform.shape == (1, 4, 4), f"Expected (1, 4, 4) transform, got {predicted_transform.shape}"

        # Remove batch dimension from transform
        transform = predicted_transform.squeeze(0)  # (4, 4)

        # Apply transformation to source points
        transformed_src_points = apply_transform(src_points, transform)

        # Compute distances to nearest neighbors in target point cloud
        # For efficiency, we compute pairwise distances and take minimum
        distances = torch.cdist(transformed_src_points.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0)  # (N, M)
        min_distances = torch.min(distances, dim=1)[0]  # (N,)

        # Compute inlier mask and ratio
        inlier_mask = (min_distances <= self.threshold)
        inlier_ratio = torch.mean(inlier_mask.float())

        scores = {'inlier_ratio': inlier_ratio}
        self.add_to_buffer(scores, datapoint)
        return scores
