from typing import Tuple, List, Union, Optional
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

    def __call__(self, *args) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.choices is not None:
            degrees = random.choice(self.choices)
        else:
            degrees = random.randint(self.range[0], self.range[1] - 1)  # Right exclusive range
        transform = lambda x: TF.rotate(x, degrees)
        result = [transform(arg) for arg in args]
        if len(result) == 1:
            result = result[0]
        return result
