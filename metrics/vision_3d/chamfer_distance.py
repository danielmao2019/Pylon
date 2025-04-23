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
            y_pred: Predicted (transformed) point cloud, shape (N, 3)
            y_true: Target point cloud, shape (M, 3)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the chamfer distance value
        """
        # Input checks
        assert y_pred.dim() == 2 and y_pred.size(1) == 3, f"Expected y_pred shape (N, 3), got {y_pred.shape}"
        assert y_true.dim() == 2 and y_true.size(1) == 3, f"Expected y_true shape (M, 3), got {y_true.shape}"
        
        # Compute nearest neighbor distances from y_pred to y_true
        y_pred_expanded = y_pred.unsqueeze(1)  # (N, 1, 3)
        y_true_expanded = y_true.unsqueeze(0)  # (1, M, 3)
        dist_matrix = torch.sqrt(((y_pred_expanded - y_true_expanded) ** 2).sum(dim=2))  # (N, M)
        
        # Find nearest neighbor distances
        min_dist_pred_to_true = torch.min(dist_matrix, dim=1)[0]  # (N,)
        
        if self.bidirectional:
            min_dist_true_to_pred = torch.min(dist_matrix, dim=0)[0]  # (M,)
            chamfer_dist = torch.mean(min_dist_pred_to_true) + torch.mean(min_dist_true_to_pred)
        else:
            chamfer_dist = torch.mean(min_dist_pred_to_true)
            
        return {"chamfer_distance": chamfer_dist}
