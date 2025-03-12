from typing import Optional
import torch


def to_one_hot(y_true: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> torch.Tensor:
    """Convert integer labels to one-hot encoded tensor.
    
    This function converts masks of shape (N, H, W) to one-hot encoded tensors
    of shape (N, C, H, W).
    
    Args:
        y_true: Integer labels tensor of shape (N, H, W)
        num_classes: Number of classes
        ignore_index: Optional index to ignore in the one-hot encoding
        
    Returns:
        One-hot encoded tensor of shape (N, C, H, W), as float32
    """
    # Input checks
    assert y_true.dim() == 3, f"Expected 3D tensor (N, H, W), got {y_true.shape=}"
    assert y_true.dtype == torch.int64, f"Expected int64 mask, got {y_true.dtype=}"
    assert isinstance(num_classes, int), f"Expected int num_classes, got {type(num_classes)=}"
    
    # Create a copy of the input for modification
    y_true_copy = y_true.clone()
    
    # If ignore_index is specified, create a mask for ignored values
    if ignore_index is not None:
        ignore_mask = (y_true == ignore_index)
        # Set ignored values to 0 temporarily for one-hot encoding
        # (we will handle them later)
        y_true_copy[ignore_mask] = 0
    
    # Check all values are within valid range for one-hot encoding
    assert (y_true_copy >= 0).all() and (y_true_copy < num_classes).all(), \
        f"Values must be in range [0, {num_classes-1}], got min={y_true_copy.min().item()}, max={y_true_copy.max().item()}"
    
    # Perform one-hot encoding directly
    result = torch.nn.functional.one_hot(y_true_copy, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # If ignore_index is specified, zero out the one-hot encoding at ignored positions
    if ignore_index is not None:
        # Expand ignore_mask to all channels
        ignore_mask = ignore_mask.unsqueeze(1).expand(-1, num_classes, -1, -1)
        # Zero out the one-hot encoding at ignored positions
        result[ignore_mask] = 0
    
    # Sanity checks
    assert result.dim() == 4
    assert result.shape[1] == num_classes
    assert result.shape[0] == y_true.shape[0]
    assert result.shape[2:] == y_true.shape[1:]
    
    return result
