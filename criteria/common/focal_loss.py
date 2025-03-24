from typing import Optional, Union
import torch
import numpy as np
from criteria.wrappers import SingleTaskCriterion
from utils.input_checks import check_classification


class FocalLoss(SingleTaskCriterion):
    """
    Focal Loss for handling class imbalance in classification tasks.
    Supports both binary and multi-class classification, as well as semantic segmentation inputs.
    """

    def __init__(
        self,
        class_weights: Optional[Union[torch.Tensor, list, tuple, np.ndarray]] = None,
        gamma: float = 0.0,
    ) -> None:
        """
        Initialize Focal Loss.
        Args:
            gamma (float): Focusing parameter (default: 0.0)
            class_weights (Optional[Union[torch.Tensor, list, tuple, np.ndarray]]):
                Weights for each class. If None, all classes are weighted equally.
                Must be a 1D tensor with length matching the number of classes.
        """
        super(FocalLoss, self).__init__()
        assert isinstance(gamma, (int, float)) and gamma >= 0, "gamma must be a non-negative number"
        self.gamma = gamma

        # Convert class_weights to tensor if provided
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
        # First prepare inputs
        y_pred = self._prepare_focal_pred(y_pred)
        y_true = self._prepare_focal_true(y_true)

        # Then validate inputs
        check_classification(y_pred, y_true)

        # Convert logits to probabilities
        probs = torch.softmax(y_pred, dim=1)

        # Get probabilities of correct classes
        indices = torch.arange(y_true.size(0), device=y_true.device)
        probs_correct = probs[indices, y_true]

        # Get class weights if available
        if self.class_weights is not None:
            # Ensure class_weights matches number of classes
            assert self.class_weights.size(0) == y_pred.size(1), \
                f"class_weights length ({self.class_weights.size(0)}) must match number of classes ({y_pred.size(1)})"
            weights = self.class_weights[y_true]
        else:
            weights = 1 / y_pred.size(1)

        # Compute focal loss
        ce_loss = -torch.log(probs_correct + 1e-7)
        focal_loss_per_class = (1 - probs_correct) ** self.gamma * ce_loss
        focal_loss_unreduced = weights * focal_loss_per_class
        focal_loss = focal_loss_unreduced.mean()

        return focal_loss
