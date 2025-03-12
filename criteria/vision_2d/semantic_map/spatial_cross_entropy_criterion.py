from typing import Tuple, Optional
import torch
from criteria.wrappers.dense_prediction_criterion import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class SpatialCrossEntropyCriterion(DenseClassificationCriterion):
    """
    Criterion for spatial cross-entropy loss in semantic segmentation tasks.
    
    This criterion extends DenseClassificationCriterion by adding dynamic class weight computation
    based on the pixel distribution in each batch when no static weights are provided.
    
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
        super(SpatialCrossEntropyCriterion, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction,
            class_weights=None,  # We compute weights dynamically
        )
        self.register_buffer('class_weights', None)

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

    def _compute_class_weights(
        self,
        y_true: torch.Tensor,
        num_classes: int,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class weights based on the frequency of each class in the ground truth.
        
        Args:
            y_true: Ground truth labels tensor of shape (N, H, W)
            num_classes: Number of classes
            valid_mask: Boolean tensor of shape (N, H, W), True for valid pixels
            
        Returns:
            Tensor of class weights
        """
        with torch.no_grad():
            class_counts = torch.bincount(y_true[valid_mask].view(-1), minlength=num_classes).float()
            total_pixels = valid_mask.sum()
            # Compute inverse frequency weights
            weights = (total_pixels - class_counts) / total_pixels
            weights = torch.clamp(weights, min=0.0)  # Ensure non-negative
            weights[class_counts == 0] = 0  # Zero weight for missing classes
            if weights.sum() > 0:  # Normalize only if there are non-zero weights
                weights = weights / weights.sum()
            return weights

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for each sample in the batch.
        
        Args:
            y_pred: Predicted logits tensor of shape (N, C, H, W)
            y_true: Ground truth labels tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Get class weights
        num_classes = y_pred.shape[1]
        class_weights = self._compute_class_weights(y_true, num_classes, valid_mask)

        # Compute per-pixel cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            y_pred,
            y_true,
            weight=class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )  # (N, H, W)

        # Compute mean loss per sample
        valid_pixels_per_sample = valid_mask.sum(dim=(1, 2))  # (N,)
        return loss.sum(dim=(1, 2)) / valid_pixels_per_sample.clamp(min=1)  # (N,)
