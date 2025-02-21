from typing import Tuple, List, Union
import random
import torch
from data.transforms import BaseTransform
from data.transforms import Crop


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

    def __call__(self, *args) -> Union[torch.Tensor, List[torch.Tensor]]:
        assert all(t.shape[-2:] == args[0].shape[-2:] for t in args)
        crop_width, crop_height = self.size
        img_height, img_width = args[0].shape[-2:]
        if crop_width > img_width or crop_height > img_height:
            raise ValueError(f"Crop size {self.size} exceeds tensor dimensions {args[0].shape[-2:]}")
        x_start = random.randint(0, img_width - crop_width)
        y_start = random.randint(0, img_height - crop_height)
        transform = Crop(loc=(x_start, y_start), size=self.size)
        result = [transform(arg) for arg in args]
        if len(result) == 1:
            result = result[0]
        return result
