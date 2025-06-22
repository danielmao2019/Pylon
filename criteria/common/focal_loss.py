from typing import Optional, Union
import numpy as np
import torch
from criteria.wrappers import SingleTaskCriterion
from utils.input_checks import check_classification


class FocalLoss(SingleTaskCriterion):
    """
    Focal Loss for handling class imbalance in classification tasks.
    Supports both binary and multi-class classification, as well as semantic segmentation inputs.
    """

    def __init__(
        self,
        gamma: float = 0.0,
        class_weights: Optional[Union[torch.Tensor, list, tuple, np.ndarray]] = None,
        ignore_value: int = -1,
        **kwargs,
    ) -> None:
        """
        Initialize Focal Loss.
        Args:
            gamma (float): Focusing parameter (default: 0.0)
            class_weights (Optional[Union[torch.Tensor, list, tuple, np.ndarray]]):
                Weights for each class. If None, all classes are weighted equally.
                Must be a 1D tensor with length matching the number of classes.
            ignore_value (int): Value to ignore in loss computation (default: -1)
        """
        super(FocalLoss, self).__init__(**kwargs)

        # Initialize gamma
        assert isinstance(gamma, (int, float)) and gamma >= 0, "gamma must be a non-negative number"
        self.gamma = gamma

        # Initialize class weights
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights)
            elif isinstance(class_weights, np.ndarray):
                class_weights = torch.from_numpy(class_weights)
            assert isinstance(class_weights, torch.Tensor), "class_weights must be convertible to torch.Tensor"
            assert class_weights.ndim == 1, "class_weights must be 1D"
            self.register_buffer('class_weights', class_weights)
        else:
            self.register_buffer('class_weights', None)

        # Initialize ignore value
        self.ignore_value = ignore_value

    def _prepare_focal_pred(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Prepare predicted logits for focal loss computation.
        Args:
            y_pred (torch.Tensor): Predicted logits with shape (N, C, ...)
        Returns:
            torch.Tensor: Reshaped logits with shape (N*..., C)
        """
        # Ensure input has at least 2 dimensions
        assert y_pred.ndim >= 2, f"Expected at least 2 dimensions, got {y_pred.ndim}"

        # If input is 2D (N, C), return as is
        if y_pred.ndim == 2:
            return y_pred

        # For higher dimensions (N, C, ...), flatten all dimensions after C
        N, C = y_pred.shape[:2]
        return y_pred.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)

    def _prepare_focal_true(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Prepare ground truth labels for focal loss computation.
        Args:
            y_true (torch.Tensor): Ground truth labels with shape (N, ...)
        Returns:
            torch.Tensor: Reshaped labels with shape (N*...)
        """
        # Ensure input has at least 1 dimension
        assert y_true.ndim >= 1, f"Expected at least 1 dimension, got {y_true.ndim}"

        # If input is 1D (N,), return as is
        if y_true.ndim == 1:
            return y_true

        # For higher dimensions (N, ...), flatten all dimensions after N
        N = y_true.shape[0]
        return y_true.reshape(N, -1).reshape(-1)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        Args:
            y_pred (torch.Tensor): Predicted logits with shape (N, C, ...)
            y_true (torch.Tensor): Ground truth labels with shape (N, ...)
        Returns:
            torch.Tensor: Computed loss
        """
        # Prepare y_pred and y_true
        y_pred = self._prepare_focal_pred(y_pred)
        y_true = self._prepare_focal_true(y_true)
        assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}."

        # Handle ignore value first
        valid_mask = y_true != self.ignore_value
        if valid_mask.sum() == 0:
            raise ValueError("All pixels in target are ignored. Cannot compute loss.")

        y_pred = y_pred[valid_mask, :]
        y_true = y_true[valid_mask]

        # Validate after handling ignore values
        check_classification(y_pred, y_true)

        # Convert logits to probabilities
        probs = torch.softmax(y_pred, dim=1)
        probs_correct = probs[torch.arange(len(y_true), device=y_true.device), y_true]

        # Get class weights if available
        if self.class_weights is not None:
            # Ensure class_weights matches number of classes
            assert self.class_weights.size(0) == y_pred.size(1), \
                f"class_weights length ({self.class_weights.size(0)}) must match number of classes ({y_pred.size(1)})"
            weights = self.class_weights[y_true]
        else:
            weights = 1 / y_pred.size(1)

        # Compute focal loss with numerical stability
        ce_loss = -torch.log(probs_correct.clamp(min=1e-7))
        focal_loss_per_class = (1 - probs_correct) ** self.gamma * ce_loss
        focal_loss_unreduced = weights * focal_loss_per_class
        focal_loss = focal_loss_unreduced.mean()

        return focal_loss
