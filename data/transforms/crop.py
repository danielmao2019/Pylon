from typing import Tuple
import torch
from data.transforms import BaseTransform


class Crop(BaseTransform):

    def _call_single_(self, tensor: torch.Tensor, loc: Tuple[int, int], size: Tuple[int, int]) -> torch.Tensor:
        """
        Crops a given tensor based on the provided location and resolution.

        Args:
            tensor (torch.Tensor): The input tensor to crop. Must have at least 2 dimensions.
            loc (Tuple[int, int]): The (x, y) starting location of the crop.
            size (Tuple[int, int]): The (width, height) resolution of the crop.

        Returns:
            torch.Tensor: The cropped tensor.
        """
        assert tensor.ndim >= 2, f"Tensor must have at least 2 dimensions, but got {tensor.shape=}"
        
        # Extract crop parameters
        x_start, y_start = loc
        crop_width, crop_height = size

        # Calculate the crop boundaries
        x_end = x_start + crop_width
        y_end = y_start + crop_height

        # Validate boundaries
        assert 0 <= x_start < x_end <= tensor.size(-1), "Crop x-coordinates are out of bounds."
        assert 0 <= y_start < y_end <= tensor.size(-2), "Crop y-coordinates are out of bounds."

        # Perform cropping
        tensor = tensor[..., y_start:y_end, x_start:x_end]
        return tensor
