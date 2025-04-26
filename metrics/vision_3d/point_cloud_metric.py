import torch
from typing import Dict, Tuple
from metrics.wrappers.single_task_metric import SingleTaskMetric


class PointCloudMetric(SingleTaskMetric):
    """
    Base class for 3D point cloud metrics.

    This class provides common functionality for metrics that operate on 3D point clouds,
    such as computing distance matrices and validating inputs.
    """

    def __init__(self) -> None:
        """
        Initialize the point cloud metric.
        """
        super(PointCloudMetric, self).__init__()

    def _validate_and_prepare_inputs(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Validate and prepare inputs for point cloud metrics.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Tuple containing:
                - y_pred: Prepared predicted point cloud
                - y_true: Prepared target point cloud
                - N: Number of points in y_pred
                - M: Number of points in y_true
        """
        # Input validation
        assert y_pred.dim() == 2 and y_pred.size(1) == 3, f"Expected y_pred shape (N, 3), got {y_pred.shape}"
        assert y_true.dim() == 2 and y_true.size(1) == 3, f"Expected y_true shape (M, 3), got {y_true.shape}"

        # Get dimensions
        N, _ = y_pred.shape
        M, _ = y_true.shape

        return y_pred, y_true, N, M

    def _compute_distance_matrix(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the distance matrix between two point clouds and find nearest neighbors.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Tuple containing:
                - dist_matrix: Distance matrix of shape (N, M)
                - min_distances: Minimum distances from each point in y_pred to y_true, shape (N,)
                - nearest_indices: Indices of nearest neighbors in y_true for each point in y_pred, shape (N,)
        """
        # Reshape for computation
        y_pred_expanded = y_pred.unsqueeze(1)  # (N, 1, 3)
        y_true_expanded = y_true.unsqueeze(0)  # (1, M, 3)

        # Compute distance matrix
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=2))  # (N, M)

        # Find nearest neighbors
        min_distances, nearest_indices = torch.min(dist_matrix, dim=1)  # (N,), (N,)

        return dist_matrix, min_distances, nearest_indices

    def _remove_batch_dimension_if_needed(
        self, result: Dict[str, torch.Tensor], is_batched: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Remove batch dimension from results if inputs were unbatched.

        Args:
            result: Dictionary of results
            is_batched: Whether inputs were batched

        Returns:
            Dictionary with batch dimension removed if inputs were unbatched
        """
        if not is_batched:
            return {k: v.squeeze(0) for k, v in result.items()}
        return result