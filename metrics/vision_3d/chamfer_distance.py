from typing import Dict
import torch
from metrics.vision_3d.point_cloud_metric import PointCloudMetric


class ChamferDistance(PointCloudMetric):
    """
    Chamfer Distance metric for 3D point cloud registration.

    Chamfer Distance measures the average distance from points in one set to
    their nearest neighbors in the other set, and vice versa. It's a bidirectional
    measure that's commonly used to evaluate the quality of point cloud registration.
    """

    DIRECTION = -1  # Lower is better

    def __init__(self, bidirectional: bool = True) -> None:
        """
        Initialize the Chamfer Distance metric.

        Args:
            bidirectional: Whether to compute bidirectional (True) or unidirectional (False) chamfer distance
        """
        super(ChamferDistance, self).__init__()
        self.bidirectional = bidirectional

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Chamfer Distance between two point clouds.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the chamfer distance value
        """
        # Validate and prepare inputs
        y_pred, y_true, N, M = self._validate_and_prepare_inputs(y_pred, y_true)

        # Compute distance matrix and find nearest neighbors
        dist_matrix, min_distances, nearest_indices = self._compute_distance_matrix(y_pred, y_true)

        if self.bidirectional:
            # Compute distances from y_true to y_pred
            _, min_distances_reverse, _ = self._compute_distance_matrix(y_true, y_pred)
            chamfer_dist = torch.mean(min_distances) + torch.mean(min_distances_reverse)
        else:
            chamfer_dist = torch.mean(min_distances)

        return {"chamfer_distance": chamfer_dist}
