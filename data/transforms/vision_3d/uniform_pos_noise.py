from typing import Any
import torch
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class UniformPosNoise(BaseTransform):

    def __init__(self, min: float = -0.1, max: float = 0.1) -> None:
        self.min = min
        self.max = max

    def _call_single(
        self, pc: PointCloud, generator: torch.Generator
    ) -> PointCloud:
        assert isinstance(pc, PointCloud), f"{type(pc)=}"

        assert generator.device.type == pc.xyz.device.type, (
            f"Generator device type '{generator.device.type}' must match point cloud device type '{pc.xyz.device.type}'"
        )

        pc.xyz = pc.xyz + torch.rand(
            pc.xyz.shape, device=pc.xyz.device, generator=generator, dtype=pc.xyz.dtype
        ) * (self.max - self.min) + self.min
        return pc
