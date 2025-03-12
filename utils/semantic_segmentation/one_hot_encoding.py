from typing import Optional
import torch


def to_one_hot(y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer labels to one-hot encoded tensor.
    
    This function converts masks of shape (N, H, W) to one-hot encoded tensors
    of shape (N, C, H, W).
    
    Args:
        y_true: Integer labels tensor of shape (N, H, W)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded tensor of shape (N, C, H, W), as float32
    """
    # Input checks
    assert y_true.dim() == 3, f"Expected 3D tensor (N, H, W), got {y_true.shape=}"
    assert y_true.dtype == torch.int64, f"Expected int64 mask, got {y_true.dtype=}"
    assert isinstance(num_classes, int), f"Expected int num_classes, got {type(num_classes)=}"
    
    # Check all values are within valid range for one-hot encoding
    assert (y_true >= 0).all() and (y_true < num_classes).all(), \
        f"Values must be in range [0, {num_classes-1}], got min={y_true.min().item()}, max={y_true.max().item()}"
    
    # Perform one-hot encoding directly
    result = torch.nn.functional.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Sanity checks
    assert result.dim() == 4
    assert result.shape[1] == num_classes
    assert result.shape[0] == y_true.shape[0]
    assert result.shape[2:] == y_true.shape[1:]
    
    return result
