import torch
from data.transforms import BaseTransform


class Flip(BaseTransform):

    def __init__(self, axis: int) -> None:
        assert type(axis) == int, f"{type(axis)=}"
        self.axis = axis

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        # check self.axis it within the range of the dimensions of tensor
        # apply flip operation about self.axis
        return tensor
