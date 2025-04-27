from typing import Optional
from abc import abstractmethod
import numpy
import torch
from criteria.vision_2d.dense_prediction.base import DensePredictionCriterion
from utils.semantic_segmentation.one_hot_encoding import to_one_hot


class DenseClassificationCriterion(DensePredictionCriterion):
    """
    Base class for dense classification tasks.

    This class extends DensePredictionCriterion with functionality specific to
    classification tasks, such as:
    - Converting logits to probabilities
    - Converting labels to one-hot encoding
    - Handling class weights
    - Computing per-class losses

    Attributes:
        ignore_value (int): Value to ignore in loss computation
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
        class_weights (Optional[torch.Tensor]): Optional weights for each class
    """

    def __init__(
        self,
        ignore_value: int = 255,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor | list | tuple | numpy.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the criterion.

        Args:
            ignore_value: Value to ignore in loss computation. Defaults to 255.
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            class_weights: Optional weights for each class. Can be a sequence, numpy array or tensor.
                         Must be 1D with non-negative values. Will be normalized to sum to 1.
        """
        super(DenseClassificationCriterion, self).__init__(
            ignore_value=ignore_value,
            reduction=reduction,
            **kwargs,
        )

        # Register class weights as buffer if provided
        if class_weights is not None:
            # Convert to tensor if needed
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights)

            # Check shape
            if class_weights.ndim != 1:
                raise ValueError(f"class_weights must be 1-dimensional, got shape {class_weights.shape}")

            # Check values
            if torch.any(class_weights < 0):
                raise ValueError("class_weights cannot contain negative values")
            if torch.all(class_weights == 0):
                raise ValueError("class_weights cannot all be zero")

            # Normalize to sum to 1
            class_weights = class_weights / class_weights.sum()
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for each sample in the batch before reduction.

        This method:
        1. Converts logits to probabilities
        2. Converts labels to one-hot encoding
        3. Computes per-class losses
        4. Applies class weights if provided
        5. Reduces over classes

        Args:
            y_pred: Logits tensor of shape (N, C, H, W)
            y_true: Labels tensor of shape (N, H, W)
            valid_mask: Boolean tensor of shape (N, H, W)

        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Convert logits to probabilities
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)  # (N, C, H, W)

        # Convert labels to one-hot
        y_true = to_one_hot(y_true, y_pred.size(1), self.ignore_value)  # (N, C, H, W)

        # Unsqueeze valid mask to (N, 1, H, W)
        valid_mask = valid_mask.unsqueeze(1)  # (N, 1, H, W)

        # Compute per-class losses
        per_class_loss = self._compute_per_class_loss(y_pred, y_true, valid_mask)  # (N, C)
        # Define class weights tensor - use provided weights or uniform weighting
        num_classes = per_class_loss.size(1)
        class_weights = self.class_weights if self.class_weights is not None else \
            torch.full((num_classes,), 1.0/num_classes, device=per_class_loss.device)

        # Apply class weights
        per_class_loss = per_class_loss * class_weights.view(1, -1)
        # Sum over classes
        return per_class_loss.sum(dim=1)  # (N,)

    @abstractmethod
    def _compute_per_class_loss(
        self,
        y_pred: torch.Tensor,  # (N, C, H, W) probabilities
        y_true: torch.Tensor,  # (N, C, H, W) one-hot encoded
        valid_mask: torch.Tensor,  # (N, 1, H, W) unsqueezed valid mask
    ) -> torch.Tensor:  # (N, C)
        """
        Compute the loss for each class and sample.

        Args:
            y_pred: Predicted probabilities tensor of shape (N, C, H, W)
            y_true: One-hot encoded ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels

        Returns:
            Loss tensor of shape (N, C) containing per-class losses for each sample
        """
        raise NotImplementedError
