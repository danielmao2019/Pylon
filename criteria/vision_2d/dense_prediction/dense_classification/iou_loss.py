from typing import Optional
import torch
from criteria.wrappers.dense_prediction_criterion import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class IoULoss(DenseClassificationCriterion):
    """
    Criterion for computing IoU loss in semantic segmentation tasks.
    
    This criterion computes the Intersection over Union (IoU) loss between predicted
    class probabilities and one-hot encoded ground truth labels for each class.
    
    Attributes:
        ignore_index: Index to ignore in the loss computation.
    """

    def __init__(
        self,
        ignore_index: int = 255,
        reduction: str = 'mean',
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in the loss computation. Defaults to 255.
            reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
        """
        super(IoULoss, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to semantic segmentation.
        
        Args:
            y_pred: Predicted logits tensor of shape (N, C, H, W)
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Raises:
            ValueError: If validation fails
        """
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid pixels (not equal to ignore_index).
        
        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid pixels
        """
        valid_mask = (y_true != self.ignore_index)
        if valid_mask.sum() == 0:
            raise ValueError("All pixels in target are ignored. Cannot compute loss.")
        return valid_mask

    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU loss for each class and sample in the batch.
        
        Args:
            y_pred: Predicted probabilities tensor of shape (N, C, H, W)
            y_true: One-hot encoded ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        # Compute IoU loss per class
        intersection = torch.sum(y_pred * y_true * valid_mask, dim=(2, 3))  # (N, C)
        union = torch.sum((y_pred + y_true - y_pred * y_true) * valid_mask, dim=(2, 3))  # (N, C)
        iou_per_class = 1 - (intersection / union.clamp(min=1e-6))  # (N, C)

        return iou_per_class
