import torch
from data.transforms import BaseTransform


class Flip(BaseTransform):

    def __init__(self, axis: int) -> None:
        assert isinstance(axis, int), f"{type(axis)=}"
        self.axis = axis

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        # Ensure self.axis is within the valid range of tensor dimensions
        if not (-tensor.ndim <= self.axis < tensor.ndim):
            raise ValueError(f"Axis {self.axis} is out of bounds for tensor with {tensor.ndim} dimensions")

        # Perform flipping along the specified axis
        return torch.flip(tensor, dims=[self.axis])
