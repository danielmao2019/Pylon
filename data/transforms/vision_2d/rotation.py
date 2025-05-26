from typing import Union
import torch
import torchvision.transforms.functional as TF
from data.transforms.base_transform import BaseTransform


class Rotation(BaseTransform):

    def __init__(self, angle: Union[int, float]) -> None:
        assert isinstance(angle, (int, float)), f"{type(angle)=}"
        self.angle = angle

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Rotates the given tensor by self.angle counterclockwise.

        Args:
            tensor (torch.Tensor): A 2D or 3D tensor (CxHxW or HxW).

        Returns:
            torch.Tensor: Rotated tensor.
        """
        assert tensor.ndim >= 2, f"Tensor must have at least 2 dimensions, but got {tensor.shape=}"
        if tensor.ndim > 2:
            return TF.rotate(tensor, self.angle)
        else:
            return TF.rotate(tensor.unsqueeze(0), self.angle).squeeze(0)
