import torch
from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion
from utils.input_checks import check_depth_estimation


class DepthEstimationCriterion(DenseRegressionCriterion):
    """
    Criterion for depth estimation tasks.
    
    This criterion computes the L1 loss between predicted and ground truth depth values
    for each pixel in the image, ignoring pixels with invalid depth measurements
    (typically marked as 0).
    
    Attributes:
        ignore_index: Value to ignore in loss computation (typically 0 for invalid depths).
        reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initialize the criterion.
        
        Args:
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
        """
        super(DepthEstimationCriterion, self).__init__(
            ignore_index=0,  # Zero values represent invalid depths
            reduction=reduction
        )

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to depth estimation.
        
        Args:
            y_pred: Predicted depth tensor of shape (N, 1, H, W)
            y_true: Ground truth depth tensor of shape (N, H, W)
            
        Raises:
            ValueError: If validation fails
        """
        check_depth_estimation(y_pred=y_pred, y_true=y_true)
        
        # Validate regression values (depths should be non-negative)
        if (y_pred < 0).any():
            raise ValueError("Predicted depths must be non-negative")
        if (y_true < 0).any():
            raise ValueError("Ground truth depths must be non-negative")

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid depth values (non-zero).
        
        Args:
            y_true: Ground truth depth tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid depths
        """
        return y_true > 0

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 loss for each sample in the batch.
        
        Args:
            y_pred: Predicted depth tensor of shape (N, 1, H, W)
            y_true: Ground truth depth tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid depths
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Remove singleton channel dimension from predictions
        y_pred = y_pred.squeeze(1)  # (N, H, W)
        
        # Compute L1 loss
        diff = torch.abs(y_pred - y_true)  # (N, H, W)
        
        # Apply valid mask
        diff = diff.masked_fill(~valid_mask, 0)
        
        # Compute mean loss per sample
        valid_pixels = valid_mask.sum(dim=(1, 2))  # (N,)
        return diff.sum(dim=(1, 2)) / valid_pixels.clamp(min=1)  # (N,)
