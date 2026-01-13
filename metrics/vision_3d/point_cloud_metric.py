from typing import Tuple, Union

import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class PointCloudMetric(SingleTaskMetric):
    """
    Base class for 3D point cloud metrics.

    This class provides common functionality for metrics that operate on 3D point clouds,
    such as computing distance matrices and validating inputs.
    """

    def _validate_and_prepare_inputs(
        self, y_pred: Union[torch.Tensor, PointCloud], y_true: Union[torch.Tensor, PointCloud]
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Validate and prepare inputs for point cloud metrics.

        Args:
            y_pred: Predicted (transformed) point cloud as PointCloud or (N, 3) tensor
            y_true: Target point cloud as PointCloud or (M, 3) tensor

        Returns:
            Tuple containing:
                - y_pred: Prepared predicted point cloud
                - y_true: Prepared target point cloud
                - N: Number of points in y_pred
                - M: Number of points in y_true
        """
        if isinstance(y_pred, PointCloud):
            y_pred_tensor = y_pred.xyz
            y_pred_count = y_pred.num_points
        else:
            PointCloud.validate_xyz_tensor(xyz=y_pred)
            y_pred_tensor = y_pred
            y_pred_count = y_pred.shape[0]

        if isinstance(y_true, PointCloud):
            y_true_tensor = y_true.xyz
            y_true_count = y_true.num_points
        else:
            PointCloud.validate_xyz_tensor(xyz=y_true)
            y_true_tensor = y_true
            y_true_count = y_true.shape[0]
        return y_pred_tensor, y_true_tensor, y_pred_count, y_true_count

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
