from typing import List, Optional, Tuple
import random
import torch
import torchvision.transforms.functional as TF
from data.transforms import BaseTransform


class RandomRotation(BaseTransform):

    def __init__(self, choices: Optional[List[int]] = None, range: Optional[Tuple[int, int]] = None) -> None:
        """
        Initializes a RandomRotation transform where the rotation angle is randomly chosen.

        Args:
            choices (Optional[List[int]]): List of discrete degrees to sample from.
            range (Optional[Tuple[int, int]]): Min/max degrees to sample from, left inclusive.

        Raises:
            ValueError: If neither or both `choices` and `range` are provided.
        """
        if (choices is None and range is None) or (choices is not None and range is not None):
            raise ValueError("Exactly one of `choices` or `range` must be provided, but got both None or both set.")

        if choices is not None:
            if not (isinstance(choices, list) and all(isinstance(d, int) for d in choices)):
                raise ValueError("`choices` must be a list of integers.")

        if range is not None:
            if not (isinstance(range, tuple) and len(range) == 2 and all(isinstance(d, int) for d in range) and range[0] < range[1]):
                raise ValueError("`range` must be a tuple of two integers (min, max) with min < max.")

        self.choices = choices
        self.range = range

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies a random rotation to the tensor.

        Args:
            tensor (torch.Tensor): A 2D or 3D tensor (CxHxW or HxW).

        Returns:
            torch.Tensor: Rotated tensor.
        """
        if self.choices is not None:
            degrees = random.choice(self.choices)
        else:
            degrees = random.randint(self.range[0], self.range[1] - 1)  # Right exclusive range

        return TF.rotate(tensor, degrees)
