import torch
import torchvision
from ..base_transform import BaseTransform


class ResizeMaps(BaseTransform):
    """
    A transformation class for resizing 2D and 3D tensors. 
    This class wraps around `torchvision.transforms.Resize` to support
    tensors with different dimensions by dynamically unsqueezing and
    squeezing along the channel dimension for compatibility.

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
        super().__init__()
        self.resize = torchvision.transforms.Resize(**kwargs)
        target_size = (self.resize.size,) * 2 if isinstance(self.resize.size, int) else tuple(self.resize.size)
        
        assert isinstance(target_size, tuple), f"Expected tuple for target_size, got {type(target_size)}"
        assert len(target_size) == 2, f"Expected a tuple of length 2, got {len(target_size)}"
        assert isinstance(target_size[0], int) and isinstance(target_size[1], int), (
            f"Both elements of target_size must be integers, got {type(target_size[0])}, {type(target_size[1])}"
        )
        self.target_size = target_size

    def _call_single_(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Dispatches the resizing function based on tensor dimensions.

        Args:
            x (torch.Tensor): Input tensor.
            height (int): Target height.
            width (int): Target width.

        Returns:
            torch.Tensor: Resized tensor.
        """
        ndim = x.ndimension()
        if ndim == 2:
            return self._call_2d(x, height, width)
        elif ndim == 3:
            return self._call_3d(x, height, width)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {ndim}")

    def _call_2d(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Resizes a 2D tensor (H, W).

        Args:
            x (torch.Tensor): Input 2D tensor.
            height (int): Target height.
            width (int): Target width.

        Returns:
            torch.Tensor: Resized 2D tensor.
        """
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert x.ndimension() == 2, f"Expected a 2D tensor, got {x.ndimension()}D tensor with shape {x.shape}"
        
        x = x.unsqueeze(0)  # Add channel dimension: (1, H, W)
        x = torchvision.transforms.functional.resize(x, (height, width))
        x = x.squeeze(0)  # Remove channel dimension: (h, w)
        
        assert x.shape == self.target_size, f"Resized tensor shape mismatch: expected {self.target_size}, got {x.shape}"
        return x

    def _call_3d(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Resizes a 3D tensor (C, H, W).

        Args:
            x (torch.Tensor): Input 3D tensor.
            height (int): Target height.
            width (int): Target width.

        Returns:
            torch.Tensor: Resized 3D tensor.
        """
        assert isinstance(x, torch.Tensor), f"Expected torch.Tensor, got {type(x)}"
        assert x.ndimension() == 3, f"Expected a 3D tensor, got {x.ndimension()}D tensor with shape {x.shape}"
        
        x = torchvision.transforms.functional.resize(x, (height, width))
        return x
