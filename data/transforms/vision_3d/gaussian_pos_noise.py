from typing import Any
import torch
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class GaussianPosNoise(BaseTransform):
    """Add Gaussian noise to point cloud positions.

    Args:
        std: Standard deviation of the Gaussian noise (default: 0.01)
    """

    def __init__(self, std: float = 0.01) -> None:
        super(GaussianPosNoise, self).__init__()
        assert isinstance(std, (int, float)), f"{type(std)=}"
        assert std >= 0, f"{std=}"
        self.std = std

    def _call_single(
        self, pc: PointCloud, generator: torch.Generator
    ) -> PointCloud:
        assert isinstance(pc, PointCloud), f"{type(pc)=}"

        assert generator.device.type == pc.xyz.device.type, (
            f"Generator device type '{generator.device.type}' must match point cloud device type '{pc.xyz.device.type}'"
        )

        if self.std > 0:
            noise = torch.randn(
                pc.xyz.shape,
                device=pc.xyz.device,
                generator=generator,
                dtype=pc.xyz.dtype,
            ) * self.std
            pc.xyz = pc.xyz + noise
        return pc
