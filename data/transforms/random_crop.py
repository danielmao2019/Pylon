import torch
import random
from typing import Tuple
from data.transforms import BaseTransform


class RandomCrop(BaseTransform):

    def __init__(self, size: Tuple[int, int]) -> None:
        """
        Initializes the RandomCrop transform with a crop size.

        Args:
            size (Tuple[int, int]): The (width, height) resolution of the crop.

        Raises:
            ValueError: If size is not a valid tuple of two positive integers.
        """
        if not (isinstance(size, tuple) and len(size) == 2 and all(isinstance(i, int) and i > 0 for i in size)):
            raise ValueError(f"size must be a tuple of two positive integers, but got {size}")

        self.size = size

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Randomly crops a given tensor based on the available region.

        Args:
            tensor (torch.Tensor): The input tensor to crop. Must have at least 2 dimensions.

        Returns:
            torch.Tensor: The randomly cropped tensor.
        """
        assert tensor.ndim >= 2, f"Tensor must have at least 2 dimensions, but got {tensor.shape=}"

        # Extract the crop width and height
        crop_width, crop_height = self.size

        # Get the original image dimensions (assumed last two dimensions are H, W)
        img_height, img_width = tensor.shape[-2:]

        # Ensure the crop size is valid
        if crop_width > img_width or crop_height > img_height:
            raise ValueError(f"Crop size {self.size} exceeds tensor dimensions {tensor.shape[-2:]}")

        # Sample a random top-left corner for cropping
        x_start = random.randint(0, img_width - crop_width)
        y_start = random.randint(0, img_height - crop_height)

        # Perform cropping
        return tensor[..., y_start:y_start + crop_height, x_start:x_start + crop_width]
