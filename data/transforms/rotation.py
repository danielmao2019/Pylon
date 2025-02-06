import torch
import torchvision.transforms.functional as TF
from data.transforms import BaseTransform


class Rotation(BaseTransform):

    def __init__(self, degrees: int) -> None:
        assert isinstance(degrees, int), f"{type(degrees)=}"
        self.degrees = degrees

    def _call_single_(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Rotates the given tensor by self.degrees counterclockwise.

        Args:
            tensor (torch.Tensor): A 2D or 3D tensor (CxHxW or HxW).

        Returns:
            torch.Tensor: Rotated tensor.
        """
        assert tensor.ndim >= 2, f"Tensor must have at least 2 dimensions, but got {tensor.shape=}"
        return TF.rotate(tensor, self.degrees)
