import torch
from typing import Dict
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
        Compute the inlier ratio between two point clouds and identify inliers.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3) or (B, N, 3)
            y_true: Target point cloud, shape (M, 3) or (B, M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - inlier_ratio: The ratio of inlier points
                - inlier_mask: Boolean mask of inliers
                - inlier_indices: List of inlier indices (or list of lists for batched inputs)
        """
        # Check if inputs are batched
        is_batched = y_pred.dim() == 3
        
        # Input validation
        if is_batched:
            assert y_pred.dim() == 3 and y_pred.size(2) == 3, f"Expected y_pred shape (B, N, 3), got {y_pred.shape}"
            assert y_true.dim() == 3 and y_true.size(2) == 3, f"Expected y_true shape (B, M, 3), got {y_true.shape}"
            assert y_pred.size(0) == y_true.size(0), f"Batch sizes must match: {y_pred.size(0)} vs {y_true.size(0)}"
        else:
            assert y_pred.dim() == 2 and y_pred.size(1) == 3, f"Expected y_pred shape (N, 3), got {y_pred.shape}"
            assert y_true.dim() == 2 and y_true.size(1) == 3, f"Expected y_true shape (M, 3), got {y_true.shape}"
            # Add batch dimension for unbatched inputs
            y_pred = y_pred.unsqueeze(0)  # (1, N, 3)
            y_true = y_true.unsqueeze(0)  # (1, M, 3)
        
        # Now both cases are treated as batched
        B, N, _ = y_pred.shape
        
        # Reshape for batched computation
        y_pred_expanded = y_pred.unsqueeze(2)  # (B, N, 1, 3)
        y_true_expanded = y_true.unsqueeze(1)  # (B, 1, M, 3)
        
        # Compute distance matrix for each batch
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=3))  # (B, N, M)
        
        # Find nearest neighbor distances
        min_distances = torch.min(dist_matrix, dim=2)[0]  # (B, N)
        
        # Count inliers
        inliers = (min_distances < self.threshold).float()
        inlier_ratio_per_batch = torch.mean(inliers, dim=1)  # Average over points in each batch
        
        # Average across batches
        inlier_ratio = torch.mean(inlier_ratio_per_batch)
        
        # Identify inliers
        inlier_mask = min_distances < self.threshold  # (B, N)
        
        # Get inlier indices for each batch
        inlier_indices = []
        for b in range(B):
            batch_indices = inlier_mask[b].nonzero().squeeze().tolist()
            # Convert single index to list for consistency
            if isinstance(batch_indices, int):
                batch_indices = [batch_indices]
            inlier_indices.append(batch_indices)
        
        # If unbatched input was provided, remove the batch dimension from the result
        if not is_batched:
            inlier_ratio = inlier_ratio.squeeze(0)
            inlier_mask = inlier_mask.squeeze(0)
            inlier_indices = inlier_indices[0]
            
        return {
            "inlier_ratio": inlier_ratio,
            "inlier_mask": inlier_mask,
            "inlier_indices": inlier_indices
        }
