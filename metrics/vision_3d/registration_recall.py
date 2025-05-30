from typing import Dict
import math
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


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
            y_pred: Predicted transformation matrix, shape (4, 4)
            y_true: Ground truth transformation matrix, shape (4, 4)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing rotation error, translation error, and registration success
        """
        # shape check
        assert y_pred.ndim == 2 and y_pred.shape == (4, 4), \
            f"Expected y_pred shape (4, 4), got {y_pred.shape}"
        assert y_true.ndim == 2 and y_true.shape == (4, 4), \
            f"Expected y_true shape (4, 4), got {y_true.shape}"
        # dtype check
        assert y_pred.dtype == y_true.dtype == torch.float32, \
            f"{y_pred.dtype=}, {y_true.dtype=}"

        # Extract rotation matrices (3x3) and translation vectors (3x1)
        R_pred = y_pred[:3, :3]  # (3, 3)
        t_pred = y_pred[:3, 3]   # (3,)

        R_true = y_true[:3, :3]  # (3, 3)
        t_true = y_true[:3, 3]   # (3,)

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

        return {
            "rotation_error_deg": angle_deg,
            "translation_error_m": trans_error,
            "registration_recall": success.float()
        }
