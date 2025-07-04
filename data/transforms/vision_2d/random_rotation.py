from typing import Tuple, List, Union, Optional, Any
import random
import torch
from data.transforms.base_transform import BaseTransform
from data.transforms.vision_2d.rotation import Rotation


class RandomRotation(BaseTransform):

    def __init__(
        self,
        choices: Optional[List[Union[int, float]]] = None,
        range: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    ) -> None:
        """
        Initializes a RandomRotation transform where the rotation angle is randomly chosen.

        Args:
            choices (Optional[List[Union[int, float]]]): List of discrete angle to sample from.
            range (Optional[Tuple[Union[int, float], Union[int, float]]]): Min/max angle to sample from, inclusive.

        Raises:
            ValueError: If neither or both `choices` and `range` are provided.
        """
        if (choices is None and range is None) or (choices is not None and range is not None):
            raise ValueError("Exactly one of `choices` or `range` must be provided, but got both None or both set.")

        if choices is not None:
            if not (isinstance(choices, list) and all(isinstance(d, (int, float)) for d in choices)):
                raise ValueError("`choices` must be a list of integers.")

        if range is not None:
            if not (isinstance(range, tuple) and len(range) == 2 and all(isinstance(d, (int, float)) for d in range) and range[0] < range[1]):
                raise ValueError("`range` must be a tuple of two integers (min, max) with min < max.")

        self.choices = choices
        self.range = range

    def __call__(self, *args, seed: Optional[Any] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        generator = self._get_generator(g_type='random', seed=seed)
        if self.choices is not None:
            angle = generator.choice(self.choices)
        else:
            angle = generator.randint(self.range[0], self.range[1] - 1)  # Right exclusive range
        angle = float(angle)
        transform = Rotation(angle)
        result = [transform(arg) for arg in args]
        if len(result) == 1:
            result = result[0]
        return result
