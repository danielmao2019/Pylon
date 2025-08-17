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
    """

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
        Applies the resizing operation to a single tensor.

        Handles tensors with shape:
        - (H, W): Resizes after unsqueezing and squeezing dimensions.
        - (..., H, W): Resizes directly using `torchvision.transforms.Resize`.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Resized tensor.

        Raises:
            ValueError: If the input tensor has fewer than 2 dimensions.
        """
        if x.ndim < 2:
            raise ValueError(f"Unsupported tensor dimensions: {x.ndim}. Expected at least 2D tensors.")
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
        x = x.unsqueeze(-3)
        x = resize_op(x)
        x = x.squeeze(-3)
        # sanity check
        assert isinstance(resize_op.size, tuple)
        assert len(resize_op.size) == 2
        assert all(isinstance(s, int) for s in resize_op.size)
        expected_shape = (*x.shape[:-2], *resize_op.size)  # (..., target_H, target_W)
        assert x.shape == expected_shape, f"Resized tensor shape mismatch: expected {expected_shape}, got {x.shape}"
        # return
        return x
