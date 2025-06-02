from typing import Tuple, Optional
import torch
from data.transforms.base_transform import BaseTransform
from data.transforms.vision_2d.resize.maps import ResizeMaps


class Crop(BaseTransform):

    def __init__(
        self,
        loc: Tuple[int, int],
        size: Tuple[int, int],
        resize: Optional[Tuple[int, int]] = None,
        interpolation: Optional[str] = None,
    ) -> None:
        """
        Initializes the Crop transform with a location and resolution.

        Args:
            loc (Tuple[int, int]): The (x, y) starting location of the crop.
            size (Tuple[int, int]): The (width, height) resolution of the crop.
            resize (Optional[Tuple[int, int]]): Target size to resize the cropped tensor. Default is None.
            interpolation (Optional[str]): Interpolation mode for resizing. Default is None.

        Raises:
            ValueError: If loc or size are not valid tuples of two positive integers.
        """
        if not (isinstance(loc, tuple) and len(loc) == 2 and all(isinstance(i, int) and i >= 0 for i in loc)):
            raise ValueError(f"loc must be a tuple of two non-negative integers, but got {loc}")
        if not (isinstance(size, tuple) and len(size) == 2 and all(isinstance(i, int) and i > 0 for i in size)):
            raise ValueError(f"size must be a tuple of two positive integers, but got {size}")

        self.loc = loc
        self.size = size
        if resize:
            self.resize_op = ResizeMaps(
                size=resize, interpolation=interpolation,
            )
        else:
            self.resize_op = None

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Crops a given tensor based on the provided location and resolution.

        Args:
            tensor (torch.Tensor): The input tensor to crop. Must have at least 2 spatial dimensions.

        Returns:
            torch.Tensor: The cropped (and possibly resized) tensor.
        """
        assert tensor.ndim >= 2, f"Tensor must have at least 2 dimensions, but got {tensor.shape=}"

        # Extract crop parameters
        x_start, y_start = self.loc
        crop_width, crop_height = self.size

        # Calculate the crop boundaries
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        # Validate boundaries
        assert 0 <= x_start < x_end <= tensor.size(-1), "Crop x-coordinates are out of bounds."
        assert 0 <= y_start < y_end <= tensor.size(-2), "Crop y-coordinates are out of bounds."

        # Perform cropping
        tensor = tensor[..., y_start:y_end, x_start:x_end]

        # Perform resizing if applicable
        if self.resize_op:
            tensor = self.resize_op(tensor)

        return tensor
