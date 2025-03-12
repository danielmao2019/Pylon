from typing import Optional, Union
from abc import abstractmethod
import torch
from criteria.vision_2d.dense_prediction.base import DensePredictionCriterion


class DenseRegressionCriterion(DensePredictionCriterion):
    """
    Base class for dense regression tasks.
    
    This class extends DensePredictionCriterion with functionality specific to
    regression tasks, such as:
    - Handling continuous target values
    - Computing per-channel losses
    - Optional normalization of predictions and targets
    
    Attributes:
        ignore_value (float): Value to ignore in loss computation
        reduction (str): How to reduce the loss over the batch dimension ('mean' or 'sum')
        normalize_inputs (bool): Whether to normalize predictions and targets
    """

    def __init__(
        self,
        ignore_value: float = float('inf'),
        reduction: str = 'mean',
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_value: Value to ignore in loss computation. Defaults to inf.
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            normalize_inputs: Whether to normalize predictions and targets before computing loss.
        """
        super(DenseRegressionCriterion, self).__init__(
            ignore_value=ignore_value,
            reduction=reduction,
        )

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for each sample in the batch before reduction.

        Args:
            y_pred: Prediction tensor of shape (N, C, H, W)
            y_true: Ground truth tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        raise NotImplementedError
