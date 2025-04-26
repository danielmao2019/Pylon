import torch
from typing import Dict
from metrics.vision_3d.point_cloud_metric import PointCloudMetric


class RMSE(PointCloudMetric):
    """
    Root Mean Square Error (RMSE) metric for 3D point cloud registration.

    RMSE measures the average magnitude of errors between corresponding points.
    For each point in the predicted (transformed) cloud, we find the nearest neighbor
    in the target cloud and compute the square root of the mean of squared distances.
    """

    DIRECTION = -1  # Lower is better

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the Root Mean Square Error between two point clouds and find correspondences.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - rmse: The root mean square error value
                - correspondences: Indices of nearest neighbors in the target cloud
        """
        # Validate and prepare inputs
        y_pred, y_true, N, M = self._validate_and_prepare_inputs(y_pred, y_true)
        
        # Compute distance matrix and find nearest neighbors
        dist_matrix, min_distances, nearest_indices = self._compute_distance_matrix(y_pred, y_true)
        
        # Compute RMSE
        rmse = torch.sqrt(torch.mean(min_distances ** 2))
            
        return {
            "rmse": rmse,
            "correspondences": nearest_indices
        }
