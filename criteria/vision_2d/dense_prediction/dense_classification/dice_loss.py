from typing import Optional
import torch
import torch.nn.functional as F
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class DiceLoss(DenseClassificationCriterion):
    """
    Criterion for computing Dice loss.
    
    This criterion computes the Dice loss between predicted class probabilities
    and ground truth labels for each pixel in the image.
    
    The Dice loss is defined as 1 - 2|X∩Y|/(|X|+|Y|) where X and Y are the
    predicted and ground truth segmentation masks.
    
    Attributes:
        ignore_index: Index to ignore in loss computation (usually background/unlabeled pixels).
        class_weights: Optional weights for each class (registered as buffer).
        reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
    """

    def __init__(
        self,
        ignore_index: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Index to ignore in loss computation (usually background/unlabeled pixels).
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
        """
        super(DiceLoss, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction,
            class_weights=class_weights
        )

    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Validate inputs specific to Dice loss.
        
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
        Compute Dice loss for each class and sample in the batch.
        
        Args:
            y_pred: Predicted probabilities tensor of shape (N, C, H, W)
            y_true: One-hot encoded ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        # Compute intersection and cardinalities
        intersection = torch.sum(y_pred * y_true * valid_mask, dim=(2, 3))  # (N, C)
        cardinality_pred = torch.sum(y_pred * valid_mask, dim=(2, 3))  # (N, C)
        cardinality_true = torch.sum(y_true * valid_mask, dim=(2, 3))  # (N, C)
        
        # Compute Dice coefficient
        dice = (2 * intersection) / (cardinality_pred + cardinality_true).clamp(min=1e-6)
        
        # Return Dice loss
        return 1 - dice
