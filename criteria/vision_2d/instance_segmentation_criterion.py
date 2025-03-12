import torch
from criteria.wrappers import DenseRegressionCriterion
from utils.input_checks import check_instance_segmentation


class InstanceSegmentationCriterion(DenseRegressionCriterion):
    """
    Criterion for instance segmentation tasks.
    
    This criterion computes the L1 loss between predicted and ground truth instance IDs
    for each pixel in the image, ignoring pixels marked with ignore_index (typically
    background or unlabeled regions).
    
    Attributes:
        ignore_index: Index to ignore in loss computation (typically background/unlabeled regions).
        reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
    """

    def __init__(self, ignore_index: int, reduction: str = 'mean') -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in loss computation (typically background
                        or unlabeled regions).
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
        """
        super(InstanceSegmentationCriterion, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction
        )

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to instance segmentation.
        
        Args:
            y_pred: Predicted instance IDs tensor of shape (N, H, W)
            y_true: Ground truth instance IDs tensor of shape (N, H, W)
            
        Raises:
            ValueError: If validation fails
        """
        check_instance_segmentation(y_pred=y_pred, y_true=y_true)
        
        # Instance IDs should be non-negative
        if (y_pred < 0).any():
            raise ValueError("Predicted instance IDs must be non-negative")
        if (y_true < 0).any():
            raise ValueError("Ground truth instance IDs must be non-negative")

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid instance IDs (not equal to ignore_index).
        
        Args:
            y_true: Ground truth instance IDs tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid instance IDs
        """
        return y_true != self.ignore_index

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 loss for each sample in the batch.
        
        Args:
            y_pred: Predicted instance IDs tensor of shape (N, H, W)
            y_true: Ground truth instance IDs tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid instance IDs
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Compute L1 loss
        diff = torch.abs(y_pred - y_true)  # (N, H, W)
        
        # Apply valid mask
        diff = diff.masked_fill(~valid_mask, 0)
        
        # Compute mean loss per sample
        valid_pixels = valid_mask.sum(dim=(1, 2))  # (N,)
        return diff.sum(dim=(1, 2)) / valid_pixels.clamp(min=1)  # (N,)
