import torch
from typing import Dict
from metrics.wrappers.single_task_metric import SingleTaskMetric


class ChamferDistance(SingleTaskMetric):
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
            y_pred: Predicted (transformed) point cloud, shape (N, 3) or (B, N, 3)
            y_true: Target point cloud, shape (M, 3) or (B, M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the chamfer distance value
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
        _, M, _ = y_true.shape
        
        # Reshape for batched computation
        y_pred_expanded = y_pred.unsqueeze(2)  # (B, N, 1, 3)
        y_true_expanded = y_true.unsqueeze(1)  # (B, 1, M, 3)
        
        # Compute distance matrix for each batch
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=3))  # (B, N, M)
        
        # Find nearest neighbor distances
        min_dist_pred_to_true = torch.min(dist_matrix, dim=2)[0]  # (B, N)
        
        if self.bidirectional:
            min_dist_true_to_pred = torch.min(dist_matrix, dim=1)[0]  # (B, M)
            chamfer_dist_per_batch = torch.mean(min_dist_pred_to_true, dim=1) + torch.mean(min_dist_true_to_pred, dim=1)
        else:
            chamfer_dist_per_batch = torch.mean(min_dist_pred_to_true, dim=1)
        
        # Average across batches
        chamfer_dist = torch.mean(chamfer_dist_per_batch)
        
        # If unbatched input was provided, remove the batch dimension from the result
        if not is_batched:
            chamfer_dist = chamfer_dist.squeeze(0)

        return {"chamfer_distance": chamfer_dist}
