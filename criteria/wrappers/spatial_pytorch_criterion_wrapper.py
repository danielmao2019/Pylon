from typing import Callable
import torch
from criteria.wrappers import SingleTaskCriterion


class SpatialPyTorchCriterionWrapper(SingleTaskCriterion):
    """
    Wrapper for PyTorch criterion that handles spatial dimensions.

    This wrapper extends SingleTaskCriterion to work with standard PyTorch loss functions
    that operate on spatial data. It handles resolution matching between predictions and ground truth.
    """

    def __init__(
        self,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> None:
        """
        Initialize the criterion wrapper.

        Args:
            criterion: PyTorch criterion to wrap
        """
        super(SpatialPyTorchCriterionWrapper, self).__init__(**kwargs)
        assert callable(criterion), f"criterion must be callable, got {type(criterion)}"
        self.register_module('criterion', criterion)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss with spatial resolution matching.

        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor

        Returns:
            Scalar loss tensor
        """
        assert isinstance(y_pred, torch.Tensor)
        assert isinstance(y_true, torch.Tensor)
        assert y_pred.ndim >= 2
        assert y_true.ndim >= 2

        # Match resolution if needed
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_true = torch.nn.functional.interpolate(y_true, size=y_pred.shape[-2:], mode='nearest')

        # Apply the criterion
        return self.criterion(input=y_pred, target=y_true)
