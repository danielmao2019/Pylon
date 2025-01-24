from typing import Tuple
import torch
from data.transforms import BaseTransform


class Crop(BaseTransform):

    def _call_single_(self, tensor: torch.Tensor, loc: Tuple[int, int], size: Tuple[int, int]) -> torch.Tensor:
        assert tensor.ndim >= 2, f"{tensor.shape=}"
        x_start = 
        x_end = 
        y_start = 
        y_end = 
        assert 0 <= x_start <= x_end <= tensor.size(-1)
        assert 0 <= y_start <= y_end <= tensor.size(-2)
        tensor = tensor[]
        tensor = tensor[]
        return tensor
