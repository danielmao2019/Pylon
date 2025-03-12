from typing import Tuple, Optional
import torch
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from utils.input_checks import check_semantic_segmentation


class SpatialCrossEntropyCriterion(DenseClassificationCriterion):
    """
    Criterion for spatial cross-entropy loss in semantic segmentation tasks.
    
    This criterion extends DenseClassificationCriterion by adding dynamic class weight computation
    based on the pixel distribution in each batch when no static weights are provided.
    
    Attributes:
        ignore_value: Value to ignore in the loss computation.
    """

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

    @torch.no_grad()
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
        class_counts = torch.bincount(y_true[valid_mask].view(-1), minlength=num_classes).float()
        total_pixels = valid_mask.sum()
        # Compute inverse frequency weights
        weights = (total_pixels - class_counts) / total_pixels
        weights = torch.clamp(weights, min=0.0)  # Ensure non-negative
        weights[class_counts == 0] = 0  # Zero weight for missing classes
        if weights.sum() > 0:  # Normalize only if there are non-zero weights
            weights = weights / weights.sum()
        return weights

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
        # Convert probabilities to log probabilities
        log_probs = torch.log(y_pred.clamp(min=1e-6))  # (N, C, H, W)

        # Compute cross entropy loss per class
        # Sum over spatial dimensions for each class
        ce_per_class = -torch.sum(y_true * log_probs * valid_mask, dim=(2, 3))  # (N, C)

        # Normalize by number of valid pixels per sample
        valid_pixels_per_sample = valid_mask.squeeze(1).sum(dim=(1, 2))  # (N,)
        ce_per_class = ce_per_class / valid_pixels_per_sample.unsqueeze(1).clamp(min=1)  # (N, C)

        # Update class weights dynamically based on current batch
        class_weights = self._compute_class_weights(
            y_true.argmax(dim=1),  # Convert one-hot back to class indices
            y_pred.shape[1],  # num_classes
            valid_mask.squeeze(1)
        )
        self.register_buffer('class_weights', class_weights)

        return ce_per_class
