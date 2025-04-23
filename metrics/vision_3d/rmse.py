import torch
from typing import Dict, Tuple
from metrics.wrappers.single_task_metric import SingleTaskMetric


class RMSE(SingleTaskMetric):
    """
    Root Mean Square Error (RMSE) metric for 3D point cloud registration.

    RMSE measures the average magnitude of errors between corresponding points.
    For each point in the predicted (transformed) cloud, we find the nearest neighbor
    in the target cloud and compute the square root of the mean of squared distances.
    """

    DIRECTION = -1  # Lower is better

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the Root Mean Square Error between two point clouds.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3) or (B, N, 3)
            y_true: Target point cloud, shape (M, 3) or (B, M, 3)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the RMSE value
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
        
        # Compute RMSE for each batch
        rmse_per_batch = torch.sqrt(torch.mean(min_distances ** 2, dim=1))  # (B,)
        
        # Average across batches
        rmse = torch.mean(rmse_per_batch)
        
        # If unbatched input was provided, remove the batch dimension from the result
        if not is_batched:
            rmse = rmse.squeeze(0)

        return {"rmse": rmse}

    def compute_with_correspondences(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RMSE and return the correspondences used for calculation.

        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3) or (B, N, 3)
            y_true: Target point cloud, shape (M, 3) or (B, M, 3)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The RMSE value and the indices of nearest neighbors
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
        
        # Find nearest neighbor distances and indices
        min_distances, indices = torch.min(dist_matrix, dim=2)  # (B, N), (B, N)
        
        # Compute RMSE for each batch
        rmse_per_batch = torch.sqrt(torch.mean(min_distances ** 2, dim=1))  # (B,)
        
        # Average across batches
        rmse = torch.mean(rmse_per_batch)
        
        # If unbatched input was provided, remove the batch dimension from the result
        if not is_batched:
            rmse = rmse.squeeze(0)
            indices = indices.squeeze(0)
            
        return rmse, indices
