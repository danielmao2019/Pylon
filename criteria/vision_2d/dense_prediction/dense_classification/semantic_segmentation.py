from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class SemanticSegmentationCriterion(DenseClassificationCriterion):
    """
    Criterion for semantic segmentation tasks.
    
    This criterion computes the cross-entropy loss between predicted class logits
    and ground truth labels for each pixel in the image.
    
    Attributes:
        ignore_value: Value to ignore in loss computation (usually background/unlabeled pixels).
        class_weights: Optional weights for each class (registered as buffer).
        reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
    """

    def __init__(
        self,
        ignore_value: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_value: Value to ignore in loss computation (usually background/unlabeled pixels).
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            class_weights: Optional weights for each class to address class imbalance.
                         Weights will be normalized to sum to 1 and must be non-negative.
        """
        super(SemanticSegmentationCriterion, self).__init__(
            ignore_value=ignore_value,
            reduction=reduction,
            class_weights=class_weights
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
        Get mask for valid pixels (not equal to ignore_value).
        
        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)
            
        Returns:
            Boolean tensor of shape (N, H, W), True for valid pixels
            
        Raises:
            ValueError: If all pixels in the target are ignored
        """
        valid_mask = y_true != self.ignore_value
        
        # Check if all pixels are ignored
        if not valid_mask.any():
            raise ValueError("All pixels in target are ignored")
            
        return valid_mask

    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for each class and sample in the batch.
        
        Args:
            y_pred: Predicted probabilities tensor of shape (N, C, H, W)
            y_true: One-hot encoded ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        # Compute cross entropy loss per class
        ce_per_class = -torch.sum(y_true * torch.log(y_pred.clamp(min=1e-6)) * valid_mask, dim=(2, 3))  # (N, C)
        
        # Normalize by number of valid pixels per sample
        valid_pixels_per_sample = valid_mask.squeeze(1).sum(dim=(1, 2))  # (N,)
        ce_per_class = ce_per_class / valid_pixels_per_sample.unsqueeze(1).clamp(min=1)  # (N, C)
        
        return ce_per_class
