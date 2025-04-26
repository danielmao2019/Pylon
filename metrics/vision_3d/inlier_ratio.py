import torch
from typing import Dict
from metrics.vision_3d.point_cloud_metric import PointCloudMetric


class InlierRatio(PointCloudMetric):
    """
    Inlier Ratio metric for 3D point cloud registration.

    Inlier Ratio measures the proportion of points in the predicted point cloud
    that are within a certain distance threshold of their nearest neighbors in
    the target point cloud.
    """

    DIRECTION = 1  # Higher is better

    def __init__(self, threshold: float = 0.1) -> None:
        """
        Initialize the Inlier Ratio metric.

        Args:
            threshold: Distance threshold for considering a point as an inlier
        """
        super(InlierRatio, self).__init__()
        self.threshold = threshold

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Inlier Ratio between two point clouds.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - inlier_ratio: The proportion of inlier points
                - inlier_mask: Boolean mask indicating which points are inliers
                - inlier_indices: Indices of inlier points
        """
        # Validate and prepare inputs
        y_pred, y_true, N, M = self._validate_and_prepare_inputs(y_pred, y_true)
        
        # Compute distance matrix and find nearest neighbors
        _, min_dist, _ = self._compute_distance_matrix(y_pred, y_true)
        
        # Count inliers (points within threshold)
        inlier_mask = (min_dist <= self.threshold)  # (N,)
        inlier_ratio = torch.mean(inlier_mask.float())
        
        # Get indices of inlier points
        inlier_indices = torch.where(inlier_mask)[0]
        
        return {
            "inlier_ratio": inlier_ratio,
            "inlier_mask": inlier_mask,
            "inlier_indices": inlier_indices
        }
