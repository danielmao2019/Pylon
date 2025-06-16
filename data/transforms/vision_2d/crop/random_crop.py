from typing import Tuple, List, Union, Optional, Any
import torch
from data.transforms.base_transform import BaseTransform
from data.transforms.vision_2d.crop.crop import Crop


class RandomCrop(BaseTransform):

    def __init__(
        self,
        size: Tuple[int, int],
        resize: Optional[Tuple[int, int]] = None,
        interpolation: Optional[str] = None,
    ) -> None:
        """
        Initializes the RandomCrop transform with a crop size.

        Args:
            size (Tuple[int, int]): The (width, height) resolution of the crop.
            resize (Optional[Tuple[int, int]]): Target size to resize the cropped tensor. Default is None.
            interpolation (Optional[str]): Interpolation mode for resizing. Default is None.

        Raises:
            ValueError: If size is not a valid tuple of two positive integers.
        """
        if not (isinstance(size, tuple) and len(size) == 2 and all(isinstance(i, int) and i > 0 for i in size)):
            raise ValueError(f"size must be a tuple of two positive integers, but got {size}")

        self.size = size
        self.resize = resize
        self.interpolation = interpolation

    def __call__(self, *args, seed: Optional[Any] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Randomly crops a region from the input tensor(s).

        Args:
            *args (torch.Tensor): One or more tensors to apply the random crop.
            seed (Optional[Any]): The seed to use for the random crop.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The cropped tensor(s).
        """
        if len(args) == 0:
            raise ValueError("RandomCrop requires at least one input tensor.")

        # Ensure all input tensors have the same spatial dimensions
        img_height, img_width = args[0].shape[-2:]
        if not all(t.shape[-2:] == (img_height, img_width) for t in args):
            raise ValueError("All input tensors must have the same spatial dimensions.")

        # Crop size
        crop_width, crop_height = self.size

        # Validate that crop size does not exceed input dimensions
        if crop_width > img_width or crop_height > img_height:
            raise ValueError(f"Crop size {self.size} exceeds tensor dimensions {img_width, img_height}.")

        # Sample a random top-left corner for cropping using the generator
        generator = self._get_generator(g_type='random', seed=seed)
        x_start = generator.randint(0, img_width - crop_width)
        y_start = generator.randint(0, img_height - crop_height)

        # Apply the Crop transform to all inputs
        transform = Crop(loc=(x_start, y_start), size=self.size, resize=self.resize, interpolation=self.interpolation)
        result = [transform(tensor) for tensor in args]

        # If only one tensor is passed, return it directly (not as a list)
        return result[0] if len(result) == 1 else result
