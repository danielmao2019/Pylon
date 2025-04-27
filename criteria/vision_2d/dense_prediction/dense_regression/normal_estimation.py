from typing import Optional
import torch
from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion
from utils.input_checks import check_normal_estimation


class NormalEstimationCriterion(DenseRegressionCriterion):
    """
    Criterion for normal estimation tasks.
    
    This criterion computes the cosine similarity loss between predicted and ground truth
    normal vectors for each pixel in the image, ignoring pixels marked with ignore_value
    (typically zero vectors for invalid normals).
    
    Attributes:
        ignore_value: Value to ignore in loss computation (typically 0 for invalid normals).
        reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
    """

    def __init__(
        self,
        reduction: str = 'mean',
        **kwargs,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
        """
        super(NormalEstimationCriterion, self).__init__(
            ignore_value=0,  # Zero vectors represent invalid normals
            reduction=reduction,
            **kwargs,
        )

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to normal estimation.
        
        Args:
            y_pred: Predicted normal vectors tensor of shape (N, 3, H, W)
            y_true: Ground truth normal vectors tensor of shape (N, 3, H, W)
            
        Raises:
            ValueError: If validation fails
        """
        check_normal_estimation(y_pred=y_pred, y_true=y_true)

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid normal vectors (non-zero norm).
        
        Args:
            y_true: Ground truth normal vectors tensor of shape (N, 3, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid normals
        """
        # Compute norm along channel dimension
        norms = torch.norm(y_true, p=2, dim=1)
        # Valid normals have non-zero norm
        return norms > 0

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity loss for each sample in the batch.
        
        Args:
            y_pred: Predicted normal vectors tensor of shape (N, 3, H, W)
            y_true: Ground truth normal vectors tensor of shape (N, 3, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid normals
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Compute cosine similarity (dot product of normalized vectors)
        cosine_map = torch.sum(y_pred * y_true, dim=1)  # (N, H, W)
        
        # Apply valid mask
        cosine_map = cosine_map.masked_fill(~valid_mask, 0)
        
        # Compute mean cosine similarity per sample
        valid_pixels = valid_mask.sum(dim=(1, 2))  # (N,)
        sample_cosine = cosine_map.sum(dim=(1, 2)) / valid_pixels.clamp(min=1)  # (N,)
        
        # Return 1 - cosine_similarity as the loss
        return 1 - sample_cosine
