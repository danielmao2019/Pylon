import torch
import torchvision
from data.transforms import BaseTransform


class ResizeMaps(BaseTransform):
    """
    A transformation class for resizing tensors with shape (..., H, W).
    This class extends `torchvision.transforms.Resize` by adding support
    for pure 2D tensors (H, W) via unsqueezing and squeezing operations.

    Attributes:
        resize (torchvision.transforms.Resize): An instance of Resize transformation.
        target_size (tuple): Target height and width of the resized tensor.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the ResizeMaps class.

        Args:
            **kwargs: Keyword arguments for `torchvision.transforms.Resize`.

        Raises:
            AssertionError: If the target size is not a tuple of two integers.
        """
        super(ResizeMaps, self).__init__()
        self.resize = torchvision.transforms.Resize(**kwargs)
        target_size = (self.resize.size,) * 2 if isinstance(self.resize.size, int) else tuple(self.resize.size)

        assert isinstance(target_size, tuple), f"Expected tuple for target_size, got {type(target_size)}"
        assert len(target_size) == 2, f"Expected a tuple of length 2, got {len(target_size)}"
        assert all(isinstance(dim, int) for dim in target_size), (
            f"Both elements of target_size must be integers, got {target_size}"
        )
        self.target_size = target_size

    def _call_single_(self, x: torch.Tensor) -> torch.Tensor:
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

        if x.ndim == 2:  # Special case for pure 2D tensors (H, W)
            return self._call_2d(x)
        else:  # Tensors with shape (..., H, W)
            return self._call_nd(x)

    def _call_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resizes a 2D tensor (H, W).

        Args:
            x (torch.Tensor): Input 2D tensor.

        Returns:
            torch.Tensor: Resized 2D tensor.
        """
        assert x.ndim == 2, f"Expected a 2D tensor, got {x.ndim}D tensor with shape {x.shape}"

        # Temporarily add batch and channel dimensions for resizing
        x = x.unsqueeze(0)  # Shape: (1, H, W)
        x = self.resize(x)  # Resize: Shape (1, target_H, target_W)
        x = x.squeeze(0)  # Remove batch and channel dimensions: (target_H, target_W)

        assert x.shape == self.target_size, f"Resized tensor shape mismatch: expected {self.target_size}, got {x.shape}"
        return x

    def _call_nd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resizes a tensor with shape (..., H, W).

        Args:
            x (torch.Tensor): Input tensor with shape (..., H, W).

        Returns:
            torch.Tensor: Resized tensor with shape (..., target_H, target_W).
        """
        assert x.ndim >= 3, f"Expected a tensor with shape (..., H, W), got {x.ndim}D tensor with shape {x.shape}"

        # Apply resizing directly
        x = self.resize(x)
        expected_shape = (*x.shape[:-2], *self.target_size)  # (..., target_H, target_W)

        assert x.shape == expected_shape, f"Resized tensor shape mismatch: expected {expected_shape}, got {x.shape}"
        return x
