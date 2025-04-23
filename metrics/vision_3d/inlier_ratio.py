import torch
from typing import Dict, Tuple, List
from metrics.wrappers.single_task_metric import SingleTaskMetric


class InlierRatio(SingleTaskMetric):
    """
    Inlier Ratio metric for 3D point cloud registration.

    The inlier ratio is the fraction of points in the predicted (transformed) cloud
    that have a nearest neighbor in the target cloud within a specified distance threshold.
    """

    DIRECTION = 1  # Higher is better

    def __init__(self, threshold: float = 0.02) -> None:
        """
        Initialize the Inlier Ratio metric.

        Args:
            threshold: Distance threshold for considering a point as an inlier (default: 0.02 units)
        """
        super(InlierRatio, self).__init__()
        self.threshold = threshold

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the inlier ratio between two point clouds.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the inlier ratio value
        """
        # Input checks
        assert y_pred.dim() == 2 and y_pred.size(1) == 3, f"Expected y_pred shape (N, 3), got {y_pred.shape}"
        assert y_true.dim() == 2 and y_true.size(1) == 3, f"Expected y_true shape (M, 3), got {y_true.shape}"

        # Compute nearest neighbor distances
        y_pred_expanded = y_pred.unsqueeze(1)  # (N, 1, 3)
        y_true_expanded = y_true.unsqueeze(0)  # (1, M, 3)
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=2))  # (N, M)

        # Find nearest neighbor distances
        min_distances = torch.min(dist_matrix, dim=1)[0]  # (N,)

        # Count inliers
        inliers = (min_distances < self.threshold).float()
        inlier_ratio = torch.mean(inliers)

        return {"inlier_ratio": inlier_ratio}

    def get_inliers(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Identify which points in the predicted cloud are inliers.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)

        Returns:
            Tuple[torch.Tensor, List[int]]: Boolean mask of inliers and list of inlier indices
        """
        # Compute nearest neighbor distances
        y_pred_expanded = y_pred.unsqueeze(1)  # (N, 1, 3)
        y_true_expanded = y_true.unsqueeze(0)  # (1, M, 3)
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=2))  # (N, M)

        # Find nearest neighbor distances
        min_distances = torch.min(dist_matrix, dim=1)[0]  # (N,)

        # Identify inliers
        inlier_mask = min_distances < self.threshold
        inlier_indices = inlier_mask.nonzero().squeeze().tolist()

        # Convert single index to list for consistency
        if isinstance(inlier_indices, int):
            inlier_indices = [inlier_indices]

        return inlier_mask, inlier_indices
