import torch
from typing import Dict
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
        
        # Compute distance matrix
        dist_matrix = self._compute_distance_matrix(y_pred, y_true)
        
        # Find nearest neighbor distances
        min_dist_pred_to_true = torch.min(dist_matrix, dim=1)[0]  # (N,)
        
        if self.bidirectional:
            min_dist_true_to_pred = torch.min(dist_matrix, dim=0)[0]  # (M,)
            chamfer_dist = torch.mean(min_dist_pred_to_true) + torch.mean(min_dist_true_to_pred)
        else:
            chamfer_dist = torch.mean(min_dist_pred_to_true)
        
        return {"chamfer_distance": chamfer_dist}
