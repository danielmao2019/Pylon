from typing import Tuple
import torch
import torchvision


class ResizeNormalEstimation:

    def __init__(self, target_size: Tuple[int, int]) -> None:
        assert type(target_size) == tuple, f"{type(target_size)=}"
        assert len(target_size) == 2, f"{len(target_size)=}"
        self.target_size = target_size

    def __call__(self, normal: torch.Tensor) -> torch.Tensor:
        assert len(normal.shape) == 3 and normal.shape[0] == 3, f"{normal.shape=}"
        normal = torchvision.transforms.Resize(size=self.target_size, antialias=True)(normal)
        normal = normal / torch.norm(normal, p=2, dim=0, keepdim=True)
        return normal
