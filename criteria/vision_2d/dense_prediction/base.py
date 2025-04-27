from typing import Optional, Union
from abc import abstractmethod
import torch
import torchvision.transforms.functional as F
from criteria.wrappers import SingleTaskCriterion


class DensePredictionCriterion(SingleTaskCriterion):
    """
    Base class for all dense prediction tasks that make per-pixel predictions.

    This includes tasks like semantic segmentation, instance segmentation, depth estimation,
    normal estimation, etc. - any task that produces an output for each pixel in the input image.

    The class handles common functionality like:
    - Ignore value handling for pixels that should not contribute to the loss
    - Resolution matching between predictions and ground truth
    - Basic shape validation
    - Batch-wise reduction of losses

    Attributes:
        ignore_value (Union[int, float]): Value to ignore in loss computation
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
    """

    REDUCTION_OPTIONS = ['mean', 'sum']

    def __init__(
        self,
        ignore_value: Optional[Union[int, float]] = None,
        reduction: str = 'mean',
        **kwargs,
    ) -> None:
        """
        Initialize the criterion.

        Args:
            ignore_value: Value to ignore in loss computation. If None, child classes should
                         provide a default value appropriate for their task.
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
        """
        super(DensePredictionCriterion, self).__init__(**kwargs)

        # Validate and set ignore_value
        if ignore_value is None:
            raise ValueError("Child classes must provide a default ignore_value if None is passed")
        if not isinstance(ignore_value, (int, float)):
            raise ValueError(f"ignore_value must be a number, got {type(ignore_value)}")
        self.ignore_value = ignore_value

        # Validate and set reduction
        if reduction not in self.REDUCTION_OPTIONS:
            raise ValueError(f"reduction must be one of {self.REDUCTION_OPTIONS}, got {reduction}")
        self.reduction = reduction

    def _match_resolution(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Match the spatial resolution of ground truth to prediction if needed.

        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor

        Returns:
            Ground truth tensor with spatial dimensions matching prediction
        """
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            # Choose interpolation mode based on data type
            mode = 'nearest' if y_true.dtype == torch.int64 else 'bilinear'
            y_true = F.resize(y_true, size=y_pred.shape[-2:], interpolation=getattr(F.InterpolationMode, mode.upper()))

        return y_true

    @abstractmethod
    def _task_specific_checks(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Perform task-specific input validation.

        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor

        Raises:
            ValueError: If validation fails
        """
        raise NotImplementedError

    @abstractmethod
    def _get_valid_mask(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Get a boolean mask indicating valid pixels based on task-specific logic.

        Args:
            y_true: Ground truth tensor

        Returns:
            Boolean tensor of same shape as y_true, True for valid pixels
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for each sample in the batch before reduction.

        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor
            valid_mask: Boolean mask indicating valid pixels

        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        raise NotImplementedError

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss following a standard sequence of steps.

        1. Task-specific input validation
        2. Match resolutions by resizing ground truth
        3. Get valid mask
        4. Compute unreduced loss
        5. Apply batch reduction

        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor

        Returns:
            Scalar loss tensor
        """
        # Input validation
        self._task_specific_checks(y_pred, y_true)

        # Match resolution
        y_true = self._match_resolution(y_pred, y_true)
        assert y_pred.shape[0] == y_true.shape[0], f"Batch size mismatch: y_pred {y_pred.shape[0]}, y_true {y_true.shape[0]}"
        assert y_pred.shape[-2:] == y_true.shape[-2:], f"Spatial dimensions mismatch: y_pred {y_pred.shape[-2:]}, y_true {y_true.shape[-2:]}"

        # Get valid mask
        valid_mask = self._get_valid_mask(y_true)
        assert valid_mask.shape == y_true.shape[0:1] + y_true.shape[-2:], \
            f"Invalid mask shape: expected {y_true.shape[0:1] + y_true.shape[-2:]}, got {valid_mask.shape}"

        # Compute unreduced loss (per sample)
        unreduced_loss = self._compute_unreduced_loss(y_pred, y_true, valid_mask)
        assert unreduced_loss.shape == (y_pred.shape[0],), \
            f"Unreduced loss should have shape ({y_pred.shape[0]},), got {unreduced_loss.shape}"

        # Apply reduction
        if self.reduction == 'mean':
            return unreduced_loss.mean()
        else:  # sum
            return unreduced_loss.sum()
