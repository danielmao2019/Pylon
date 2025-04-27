from typing import Optional
import torch
import torch.nn.functional as F
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class IoULoss(DenseClassificationCriterion):
    """
    IoU Loss for semantic segmentation tasks.
    
    The IoU (Intersection over Union) coefficient measures the overlap between predictions 
    and ground truth. This loss is defined as 1 - IoU.
    
    This implementation supports:
    - Standard IoU loss (1 - IoU coefficient)
    - Class weights to handle class imbalance
    - Ignore value to exclude specific pixel values from loss computation
    
    Attributes:
        ignore_value (int): Value to ignore in loss computation
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
        class_weights (Optional[torch.Tensor]): Optional weights for each class
    """

    def __init__(
        self,
        ignore_value: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_value: Value to ignore in loss computation (usually background/unlabeled pixels).
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
        """
        super(IoULoss, self).__init__(
            ignore_value=ignore_value,
            reduction=reduction,
            class_weights=class_weights,
            **kwargs,
        )

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to IoU loss.
        
        Args:
            y_pred: Predicted logits tensor of shape (N, C, H, W)
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Raises:
            ValueError: If validation fails
        """
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)

    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid pixels (not equal to ignore_value).
        
        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid pixels
        """
        valid_mask = (y_true != self.ignore_value)
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
        # Compute intersection and union
        intersection = torch.sum(y_pred * y_true * valid_mask, dim=(2, 3))  # (N, C)
        union = torch.sum((y_pred + y_true - y_pred * y_true) * valid_mask, dim=(2, 3))  # (N, C)
        
        # Compute IoU
        iou = intersection / union.clamp(min=1e-6)
        
        # Return IoU loss
        return 1 - iou
