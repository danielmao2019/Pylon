import torch
import math
from typing import Dict, List, Tuple, Union, Optional
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_write_file
from utils.io import save_json


class RegistrationRecall(SingleTaskMetric):
    """
    Registration Recall metric for 3D point cloud registration.
    
    This metric computes rotation and translation errors between estimated and ground truth
    transformations, and evaluates registration recall (percentage of successful registrations).
    """

    DIRECTION = 1  # Higher is better
    
    def __init__(self, rot_threshold_deg: float = 5.0, trans_threshold_m: float = 0.3) -> None:
        """
        Initialize the Registration Recall metric.
        
        Args:
            rot_threshold_deg: Rotation error threshold in degrees for considering a registration successful
            trans_threshold_m: Translation error threshold in meters (or units) for considering a registration successful
        """
        super(RegistrationRecall, self).__init__()
        self.rot_threshold_deg = rot_threshold_deg
        self.trans_threshold_m = trans_threshold_m
    
    def _compute_score(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute registration error metrics.
        
        Args:
            y_pred: Predicted transformation matrix, shape (4, 4) or (B, 4, 4) for batch
            y_true: Ground truth transformation matrix, shape (4, 4) or (B, 4, 4) for batch
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing rotation error, translation error, and registration success
        """
        # Handle batched or single input
        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(0)  # (1, 4, 4)
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(0)  # (1, 4, 4)
        
        # Input checks
        assert y_pred.dim() == 3 and y_pred.size(1) == 4 and y_pred.size(2) == 4, \
            f"Expected y_pred shape (B, 4, 4), got {y_pred.shape}"
        assert y_true.dim() == 3 and y_true.size(1) == 4 and y_true.size(2) == 4, \
            f"Expected y_true shape (B, 4, 4), got {y_true.shape}"
        assert y_pred.size(0) == y_true.size(0), \
            f"Batch size mismatch: {y_pred.size(0)} vs {y_true.size(0)}"
        
        batch_size = y_pred.size(0)
        rotation_errors = []
        translation_errors = []
        registration_successes = []
        
        for i in range(batch_size):
            # Extract rotation matrices (3x3) and translation vectors (3x1)
            R_pred = y_pred[i, :3, :3]  # (3, 3)
            t_pred = y_pred[i, :3, 3]   # (3,)
            
            R_true = y_true[i, :3, :3]  # (3, 3)
            t_true = y_true[i, :3, 3]   # (3,)
            
            # Compute relative rotation error
            R_error = torch.matmul(R_true.transpose(0, 1), R_pred)  # R_true^T * R_pred
            
            # Convert to rotation angle in degrees
            # The formula is: angle = arccos((trace(R_error) - 1) / 2)
            rot_trace = R_error.trace()
            rot_trace = torch.clamp(rot_trace, min=-1.0, max=3.0)  # Clamp to avoid numerical issues
            angle_rad = torch.acos((rot_trace - 1) / 2)
            angle_deg = angle_rad * 180 / math.pi
            
            # Compute translation error (Euclidean distance)
            trans_error = torch.norm(t_pred - t_true)
            
            # Determine if registration is successful
            success = (angle_deg < self.rot_threshold_deg) & (trans_error < self.trans_threshold_m)
            
            rotation_errors.append(angle_deg)
            translation_errors.append(trans_error)
            registration_successes.append(success.float())
        
        # Convert lists to tensors
        rotation_errors = torch.stack(rotation_errors)
        translation_errors = torch.stack(translation_errors)
        registration_successes = torch.stack(registration_successes)
        
        # Computing registration recall (percentage of successful registrations)
        recall = torch.mean(registration_successes)
        
        return {
            "rotation_error_deg": torch.mean(rotation_errors),
            "translation_error_m": torch.mean(translation_errors),
            "registration_recall": recall
        }
    
    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        """
        Summarize accumulated scores.
        
        Args:
            output_path: Path to save the results, if provided.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of summarized metric scores.
        """
        result = super(RegistrationRecall, self).summarize(output_path)
        return result 