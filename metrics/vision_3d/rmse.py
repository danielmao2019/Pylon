import torch
from typing import Dict, Tuple, Union, Optional
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_write_file
from utils.io import save_json


class RMSE(SingleTaskMetric):
    """
    Root Mean Square Error (RMSE) metric for 3D point cloud registration.
    
    RMSE measures the average magnitude of errors between corresponding points.
    For each point in the predicted (transformed) cloud, we find the nearest neighbor
    in the target cloud and compute the square root of the mean of squared distances.
    """

    DIRECTION = -1  # Lower is better
    
    def __init__(self) -> None:
        """
        Initialize the RMSE metric.
        """
        super(RMSE, self).__init__()
    
    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the Root Mean Square Error between two point clouds.
        
        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the RMSE value
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
        
        # Compute RMSE
        rmse = torch.sqrt(torch.mean(min_distances ** 2))
        
        return {"rmse": rmse}
    
    def compute_with_correspondences(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RMSE and return the correspondences used for calculation.
        
        Args:
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The RMSE value and the indices of nearest neighbors
        """
        # Compute nearest neighbor distances
        y_pred_expanded = y_pred.unsqueeze(1)  # (N, 1, 3)
        y_true_expanded = y_true.unsqueeze(0)  # (1, M, 3)
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=2))  # (N, M)
        
        # Find nearest neighbor distances and indices
        min_distances, indices = torch.min(dist_matrix, dim=1)  # (N,), (N,)
        
        # Compute RMSE
        rmse = torch.sqrt(torch.mean(min_distances ** 2))
        
        return rmse, indices 