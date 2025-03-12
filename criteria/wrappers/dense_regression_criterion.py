from typing import Optional, Tuple, Union
import torch
from criteria.wrappers.dense_prediction_criterion import DensePredictionCriterion


class DenseRegressionCriterion(DensePredictionCriterion):
    """
    Base class for dense regression tasks that predict continuous values for each pixel.
    
    This includes tasks like:
    - Depth estimation (scalar value per pixel)
    - Normal estimation (3D vector per pixel)
    - Instance segmentation (continuous instance embeddings)
    
    The class handles common functionality like:
    - Validation of regression outputs
    - Handling of invalid/ignored values
    - Common regression metrics (L1, L2, etc.)
    
    Attributes:
        ignore_index: Value to ignore in loss computation (e.g., invalid measurements)
    """

    def __init__(
        self,
        ignore_index: Optional[Union[int, float]] = None,
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize the criterion.
        
        Args:
            ignore_index: Value to ignore in loss computation. If None, child classes should
                         provide a default value appropriate for their task.
            reduction: How to reduce the loss over valid pixels ('mean' or 'sum')
        """
        super(DenseRegressionCriterion, self).__init__(ignore_index=ignore_index)
        
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction
        
    def _validate_regression_values(
        self,
        tensor: torch.Tensor,
        allow_negative: bool = True,
        name: str = "tensor"
    ) -> None:
        """
        Validate regression values in a tensor.
        
        Args:
            tensor: Input tensor to validate
            allow_negative: Whether negative values are allowed
            name: Name of the tensor for error messages
            
        Raises:
            ValueError: If validation fails
        """
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Found non-finite values in {name}")
            
        if not allow_negative and (tensor < 0).any():
            raise ValueError(f"Found negative values in {name} where not allowed")
            
    def _compute_l1_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L1 (mean absolute error) loss over valid pixels.
        
        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor
            valid_mask: Boolean mask of valid pixels
            
        Returns:
            L1 loss value
        """
        diff = torch.abs(y_pred - y_true)
        
        # Apply mask
        if valid_mask is not None:
            diff = diff[valid_mask]
            
        # Reduce
        if self.reduction == 'mean':
            return diff.mean()
        else:
            return diff.sum()
            
    def _compute_l2_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 (mean squared error) loss over valid pixels.
        
        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor
            valid_mask: Boolean mask of valid pixels
            
        Returns:
            L2 loss value
        """
        diff = torch.square(y_pred - y_true)
        
        # Apply mask
        if valid_mask is not None:
            diff = diff[valid_mask]
            
        # Reduce
        if self.reduction == 'mean':
            return diff.mean()
        else:
            return diff.sum()
            
    def _compute_smooth_l1_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        valid_mask: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Smooth L1 (Huber) loss over valid pixels.
        
        Args:
            y_pred: Prediction tensor
            y_true: Ground truth tensor
            valid_mask: Boolean mask of valid pixels
            beta: Threshold for switching between L1 and L2 loss
            
        Returns:
            Smooth L1 loss value
        """
        diff = torch.abs(y_pred - y_true)
        
        # Apply mask
        if valid_mask is not None:
            diff = diff[valid_mask]
            
        # Compute loss
        loss = torch.where(
            diff < beta,
            0.5 * diff.pow(2) / beta,
            diff - 0.5 * beta
        )
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum() 