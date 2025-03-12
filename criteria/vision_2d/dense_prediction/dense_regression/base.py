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
        normalize_inputs: bool = False,
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_value: Value to ignore in loss computation. Defaults to inf.
            reduction: How to reduce the loss over the batch dimension ('mean' or 'sum').
            normalize_inputs: Whether to normalize predictions and targets before computing loss.
        """
        super(DenseRegressionCriterion, self).__init__(
            ignore_index=ignore_value,
            reduction=reduction,
        )
        self.normalize_inputs = normalize_inputs

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor to zero mean and unit variance.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Normalized tensor of same shape
        """
        if not self.normalize_inputs:
            return x
            
        # Compute statistics over spatial dimensions only
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-8)

    def _compute_unreduced_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for each sample in the batch before reduction.
        
        This method:
        1. Optionally normalizes inputs
        2. Computes per-channel losses
        3. Reduces over channels
        
        Args:
            y_pred: Prediction tensor of shape (N, C, H, W)
            y_true: Target tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W)
            
        Returns:
            Loss tensor of shape (N,) containing per-sample losses
        """
        # Normalize inputs if requested
        if self.normalize_inputs:
            y_pred = self._normalize(y_pred)
            y_true = self._normalize(y_true)
        
        # Compute per-channel losses
        per_channel_loss = self._compute_per_channel_loss(y_pred, y_true, valid_mask)  # (N, C)
        
        # Sum over channels
        return per_channel_loss.sum(dim=1)  # (N,)

    @abstractmethod
    def _compute_per_channel_loss(
        self,
        y_pred: torch.Tensor,  # (N, C, H, W)
        y_true: torch.Tensor,  # (N, C, H, W)
        valid_mask: torch.Tensor,  # (N, 1, H, W)
    ) -> torch.Tensor:  # (N, C)
        """
        Compute the loss for each channel and sample.
        
        Args:
            y_pred: Prediction tensor of shape (N, C, H, W)
            y_true: Target tensor of shape (N, C, H, W)
            valid_mask: Boolean tensor of shape (N, 1, H, W), True for valid pixels
            
        Returns:
            Loss tensor of shape (N, C) containing per-channel losses for each sample
        """
        raise NotImplementedError 