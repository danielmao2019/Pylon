from typing import Dict
import torch
from metrics.vision_3d.point_cloud_metric import PointCloudMetric


class MAE(PointCloudMetric):
    """
    Mean Absolute Error (MAE) metric for 3D point cloud registration.

    MAE measures the average absolute difference between corresponding points.
    For each point in the predicted (transformed) cloud, we find the nearest neighbor
    in the target cloud and compute the absolute distance.
    """

    DIRECTION = -1  # Lower is better

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the Mean Absolute Error between two point clouds.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the MAE value
        """
        # Validate and prepare inputs
        y_pred, y_true, N, M = self._validate_and_prepare_inputs(y_pred, y_true)

        # Compute distance matrix and find nearest neighbors
        dist_matrix, min_distances, nearest_indices = self._compute_distance_matrix(y_pred, y_true)

        # Compute mean absolute error
        mae = torch.mean(min_distances)

        return {"mae": mae}
