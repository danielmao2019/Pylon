from typing import Dict, Any, Optional
import torch
from metrics.base_metric import BaseMetric
from utils.input_checks.str_types import check_write_file
from utils.io.json import safe_save_json
from utils.ops.dict_as_tensor import transpose_buffer


class TransformInlierRatio(BaseMetric):
    """
    Inlier ratio metric for transform-based point cloud registration algorithms.

    This metric works with algorithms that directly predict transformation matrices.
    It takes source and target point clouds from inputs, predicted transform from outputs,
    and computes the inlier ratio by:
    1. Applying predicted transform to source points
    2. Computing nearest neighbor distances to target points using cdist
    3. Counting inliers within threshold

    Args:
        threshold: Distance threshold for considering a point as inlier
    """

    DIRECTIONS = {
        'inlier_ratio': +1  # Higher is better
    }

    def __init__(self, threshold: float = 0.1, use_buffer: bool = True) -> None:
        """Initialize the TransformInlierRatio metric.

        Args:
            threshold: Distance threshold for inlier determination
            use_buffer: Whether to use buffer for storing results
        """
        super(TransformInlierRatio, self).__init__(use_buffer=use_buffer)
        self.threshold = threshold

    def __call__(self, datapoint: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Compute transform-based inlier ratio.

        Args:
            datapoint: Dictionary containing:
                - inputs: {'src_pc': source points, 'tgt_pc': target points}
                - outputs: {'transform': predicted transformation matrix}
                - labels: (not used for this metric)
                - meta_info: metadata

        Returns:
            Dictionary containing inlier ratio score
        """
        # Extract inputs and outputs from datapoint
        assert 'inputs' in datapoint and 'outputs' in datapoint
        inputs = datapoint['inputs']
        outputs = datapoint['outputs']

        # Validate inputs
        assert 'src_pc' in inputs and 'tgt_pc' in inputs, f"Expected src_pc and tgt_pc in inputs, got {inputs.keys()}"

        # Handle point clouds - they may be tensors directly or dicts with 'pos' key
        src_points = inputs['src_pc']
        tgt_points = inputs['tgt_pc']

        if isinstance(src_points, dict):
            assert 'pos' in src_points, "Expected 'pos' key in src_pc dict"
            src_points = src_points['pos']
        if isinstance(tgt_points, dict):
            assert 'pos' in tgt_points, "Expected 'pos' key in tgt_pc dict"
            tgt_points = tgt_points['pos']

        # Handle transform - it may be a dict with 'transform' key or directly the tensor
        if isinstance(outputs, dict):
            assert 'transform' in outputs, f"Expected transform in outputs, got {outputs.keys()}"
            transform = outputs['transform']
        else:
            # outputs is directly the transform tensor (e.g., from ICP, RANSAC)
            transform = outputs

        # Input validation
        assert isinstance(src_points, torch.Tensor), f"Expected torch.Tensor for src_points, got {type(src_points)}"
        assert isinstance(tgt_points, torch.Tensor), f"Expected torch.Tensor for tgt_points, got {type(tgt_points)}"
        assert isinstance(transform, torch.Tensor), f"Expected torch.Tensor for transform, got {type(transform)}"

        # Handle batched point clouds - squeeze batch dimension if present
        if src_points.ndim == 3:
            assert src_points.shape[0] == 1, f"Expected batch size 1, got {src_points.shape[0]}"
            src_points = src_points.squeeze(0)
        if tgt_points.ndim == 3:
            assert tgt_points.shape[0] == 1, f"Expected batch size 1, got {tgt_points.shape[0]}"
            tgt_points = tgt_points.squeeze(0)

        assert src_points.ndim == 2 and src_points.shape[1] == 3, f"Expected src_points shape (N, 3), got {src_points.shape}"
        assert tgt_points.ndim == 2 and tgt_points.shape[1] == 3, f"Expected tgt_points shape (M, 3), got {tgt_points.shape}"

        # Handle transform shape - can be (4, 4) or (1, 4, 4)
        if transform.ndim == 3:
            assert transform.shape == (1, 4, 4), f"Expected transform shape (1, 4, 4), got {transform.shape}"
            transform = transform.squeeze(0)  # Remove batch dimension
        elif transform.ndim == 2:
            assert transform.shape == (4, 4), f"Expected transform shape (4, 4), got {transform.shape}"
        else:
            raise ValueError(f"Expected transform to be 2D or 3D tensor, got {transform.ndim}D")

        # Apply transformation to source points
        # Convert to homogeneous coordinates
        src_homogeneous = torch.cat([src_points, torch.ones(src_points.shape[0], 1, device=src_points.device)], dim=1)  # (N, 4)

        # Apply transformation: (4, 4) @ (N, 4).T = (4, N) -> (N, 4).T
        transformed_src = (transform @ src_homogeneous.T).T  # (N, 4)

        # Convert back to 3D coordinates
        transformed_src_3d = transformed_src[:, :3]  # (N, 3)

        # Compute pairwise distances and find minimum distance for each transformed source point
        distances = torch.cdist(transformed_src_3d.unsqueeze(0), tgt_points.unsqueeze(0)).squeeze(0)  # (N, M)
        min_distances, _ = torch.min(distances, dim=1)  # (N,)

        # Count inliers (points within threshold)
        inliers = (min_distances <= self.threshold).float()
        inlier_ratio = inliers.mean()

        scores = {'inlier_ratio': inlier_ratio}
        self.add_to_buffer(scores, datapoint)
        return scores

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Summarize inlier ratios across all data points."""
        assert self.use_buffer and hasattr(self, 'buffer') and self.buffer is not None
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0

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
            safe_save_json(obj=result, filepath=output_path)
        return result
