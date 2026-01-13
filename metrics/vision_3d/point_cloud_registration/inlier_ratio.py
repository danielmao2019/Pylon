from typing import Any, Dict

import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from data.structures.three_d.point_cloud.ops.apply_transform import apply_transform
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class InlierRatio(SingleTaskMetric):
    """
    Inlier Ratio metric for 3D point cloud registration.

    Inlier Ratio measures the proportion of points in the source point cloud
    that are within a certain distance threshold of their nearest neighbors in
    the target point cloud after applying the predicted transformation.
    """

    DIRECTIONS = {
        'inlier_ratio': +1  # Higher is better
    }

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
                - 'outputs': {'src_pc': predicted_source_points, 'tgt_pc': predicted_target_correspondences} as PointCloud, (N, 3) tensor, or List[PointCloud] batch
                - 'labels': {'transform': ground_truth_transform_matrix}
                - 'meta_info': {...}

        Returns:
            Dictionary containing inlier ratio score
        """
        # Extract predicted correspondences from outputs
        assert 'outputs' in datapoint, "Missing 'outputs' in datapoint"
        assert 'labels' in datapoint, "Missing 'labels' in datapoint"
        assert 'src_pc' in datapoint['outputs'] and 'tgt_pc' in datapoint['outputs'], f"Missing point clouds in outputs: {datapoint['outputs'].keys()}"
        assert 'transform' in datapoint['labels'], f"Missing transform in labels: {datapoint['labels'].keys()}"
        outputs = datapoint['outputs']
        labels = datapoint['labels']

        src_input = outputs['src_pc']  # Predicted source points
        tgt_input = outputs['tgt_pc']  # Predicted target correspondences

        def _normalize_point_cloud(pc_input: Any) -> list[PointCloud]:
            if isinstance(pc_input, list):
                assert len(pc_input) == 1, f"Expected batch size 1, got {len(pc_input)}"
                assert all(isinstance(pc, PointCloud) for pc in pc_input), f"Non-PointCloud entries in pc list: {tuple(type(pc) for pc in pc_input)}"
                return pc_input
            elif isinstance(pc_input, PointCloud):
                return [pc_input]
            elif isinstance(pc_input, torch.Tensor):
                if pc_input.ndim == 2:
                    assert pc_input.shape[1] == 3, f"Expected (N, 3) tensor, got {pc_input.shape}"
                    return [PointCloud(xyz=pc_input)]
                elif pc_input.ndim == 3:
                    assert pc_input.shape[0] == 1, f"Expected batch size 1, got {pc_input.shape[0]}"
                    assert pc_input.shape[2] == 3, f"Expected point dimensionality 3, got {pc_input.shape[2]}"
                    return [PointCloud(xyz=pc_input.squeeze(0))]
                else:
                    assert 0, f"Expected 2D or 3D tensor, got ndim={pc_input.ndim}"
            else:
                assert 0, f"Unsupported point cloud input type: {type(pc_input)}"

        src_pcs = _normalize_point_cloud(pc_input=src_input)
        tgt_pcs = _normalize_point_cloud(pc_input=tgt_input)
        assert len(src_pcs) == 1 and len(tgt_pcs) == 1, f"Expected single point cloud per batch, got {len(src_pcs)} and {len(tgt_pcs)}"

        src_pc = src_pcs[0]
        tgt_pc = tgt_pcs[0]
        outputs['src_pc'] = src_pc
        outputs['tgt_pc'] = tgt_pc
        assert src_pc.num_points == tgt_pc.num_points, f"Mismatched correspondence count: {src_pc.num_points} vs {tgt_pc.num_points}"
        src_points = src_pc.xyz
        tgt_points = tgt_pc.xyz

        # Extract ground truth transformation from labels
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
