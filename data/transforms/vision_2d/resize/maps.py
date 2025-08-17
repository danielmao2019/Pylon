from typing import Optional
import torch
import torchvision
from data.transforms.base_transform import BaseTransform


class ResizeMaps(BaseTransform):
    """
    A transformation class for resizing tensors with shape (..., H, W).
    This class extends `torchvision.transforms.Resize` by adding support
    for pure 2D tensors (H, W) via unsqueezing and squeezing operations.
    
    CRITICAL: This implementation properly handles ignore values to prevent
    interpolation corruption. When bilinear interpolation would mix ignore 
    values with valid values, it uses a masking approach to preserve data integrity.

    Attributes:
        resize (torchvision.transforms.Resize): An instance of Resize transformation.
        target_size (tuple): Target height and width of the resized tensor.
        ignore_value (Optional[float]): Value to treat as ignore/invalid during resizing.
        TOLERANCE (float): Tolerance for floating-point ignore value comparison.
    """
    
    # Class constant for tolerance-based ignore value comparison
    TOLERANCE = 1e-5

    def __init__(self, ignore_value: Optional[float] = None, **kwargs) -> None:
        """
        Initializes the ResizeMaps class.

        Args:
            ignore_value: Single value to treat as ignore/invalid during resizing.
                         Common examples: -1.0 (depth maps), 255 (segmentation masks).
                         If None, standard resizing is applied without special handling.
            **kwargs: Keyword arguments for `torchvision.transforms.Resize`.

        Raises:
            ValueError: If unsupported interpolation mode is provided.
        """
        super(ResizeMaps, self).__init__()
        
        # Store ignore value (can be None for no special handling)
        self.ignore_value = ignore_value
        
        # Handle interpolation parameter
        if 'interpolation' in kwargs:
            if kwargs['interpolation'] is None:
                pass
            elif kwargs['interpolation'] == "bilinear":
                kwargs['interpolation'] = torchvision.transforms.functional.InterpolationMode.BILINEAR
            elif kwargs['interpolation'] == "nearest":
                kwargs['interpolation'] = torchvision.transforms.functional.InterpolationMode.NEAREST
            else:
                raise ValueError(f"Unsupported interpolation mode: {kwargs['interpolation']}")
        if kwargs.get('interpolation', None):
            self.resize_op = torchvision.transforms.Resize(**kwargs)
        else:
            self.resize_op = None
            self.kwargs = kwargs

    def _call_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the resizing operation to a single tensor with proper ignore value handling.

        Handles tensors with shape:
        - (H, W): Resizes after unsqueezing and squeezing dimensions.
        - (..., H, W): Resizes directly using `torchvision.transforms.Resize`.

        CRITICAL: When ignore_value is specified and bilinear interpolation is used,
        this method prevents interpolation corruption by using a mask-based approach.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Resized tensor with properly preserved ignore values.

        Raises:
            ValueError: If the input tensor has fewer than 2 dimensions.
        """
        if x.ndim < 2:
            raise ValueError(f"Unsupported tensor dimensions: {x.ndim}. Expected at least 2D tensors.")
        
        # Get resize operation
        if not self.resize_op:
            interpolation = (
                torchvision.transforms.functional.InterpolationMode.BILINEAR
                if torch.is_floating_point(x) else
                torchvision.transforms.functional.InterpolationMode.NEAREST
            )
            self.kwargs['interpolation'] = interpolation
            resize_op = torchvision.transforms.Resize(**self.kwargs)
        else:
            resize_op = self.resize_op
        
        # Use ignore-value-aware resizing only for bilinear interpolation with ignore_value specified
        if (self.ignore_value is not None and 
            resize_op.interpolation == torchvision.transforms.functional.InterpolationMode.BILINEAR):
            return self._ignore_aware_resize(x, resize_op)
        else:
            # Standard resizing for: no ignore_value, nearest neighbor, or other interpolation modes
            return self._standard_resize(x, resize_op)
    
    def _standard_resize(self, x: torch.Tensor, resize_op: torchvision.transforms.Resize) -> torch.Tensor:
        """Apply standard resizing without ignore value handling."""
        x = x.unsqueeze(-3)
        x = resize_op(x)
        x = x.squeeze(-3)
        
        # Sanity check
        assert isinstance(resize_op.size, tuple)
        assert len(resize_op.size) == 2
        assert all(isinstance(s, int) for s in resize_op.size)
        expected_shape = (*x.shape[:-2], *resize_op.size)  # (..., target_H, target_W)
        assert x.shape == expected_shape, f"Resized tensor shape mismatch: expected {expected_shape}, got {x.shape}"
        
        return x
    
    def _ignore_aware_resize(self, x: torch.Tensor, resize_op: torchvision.transforms.Resize) -> torch.Tensor:
        """
        Apply resizing with ignore value protection for bilinear interpolation.
        
        Strategy:
        1. Create mask of valid pixels (non-ignore values)
        2. Replace ignore values with a neutral value for interpolation
        3. Resize both data and mask
        4. Restore ignore values where mask indicates invalid regions
        """
        # Create valid pixel mask using tolerance for floating-point comparison
        import math
        if math.isnan(self.ignore_value):
            valid_mask = ~torch.isnan(x)
        else:
            # Use tolerance-based comparison for floating-point ignore values
            valid_mask = torch.abs(x - self.ignore_value) >= self.TOLERANCE
        
        # If no ignore values present, use standard resizing
        if valid_mask.all():
            return self._standard_resize(x, resize_op)
        
        # Replace ignore values with mean of valid values for interpolation
        valid_pixels = x[valid_mask]
        if len(valid_pixels) > 0:
            fill_value = valid_pixels.mean().item()
        else:
            # All pixels are ignore values - use zero
            fill_value = 0.0
        
        # Create data tensor with ignore values replaced
        x_filled = x.clone()
        x_filled[~valid_mask] = fill_value
        
        # Create mask tensor (1.0 for valid, 0.0 for ignore)
        mask = valid_mask.float()
        
        # Resize both data and mask
        x_filled = x_filled.unsqueeze(-3)
        mask = mask.unsqueeze(-3)
        
        x_resized = resize_op(x_filled)
        mask_resized = resize_op(mask)
        
        x_resized = x_resized.squeeze(-3)
        mask_resized = mask_resized.squeeze(-3)
        
        # Restore ignore values where mask indicates invalid regions
        # Use threshold of 0.5 - pixels are valid if mask > 0.5
        ignore_threshold = 0.5
        ignore_regions = mask_resized <= ignore_threshold
        x_resized[ignore_regions] = self.ignore_value
        
        # Sanity check
        assert isinstance(resize_op.size, tuple)
        assert len(resize_op.size) == 2
        assert all(isinstance(s, int) for s in resize_op.size)
        expected_shape = (*x_resized.shape[:-2], *resize_op.size)  # (..., target_H, target_W)
        assert x_resized.shape == expected_shape, f"Resized tensor shape mismatch: expected {expected_shape}, got {x_resized.shape}"
        
        return x_resized
