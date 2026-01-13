from typing import Any
import torch
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.select import Select
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class Shuffle(BaseTransform):

    def _call_single(
        self, pc: PointCloud, generator: torch.Generator
    ) -> PointCloud:
        assert isinstance(pc, PointCloud), f"{type(pc)=}"

        assert generator.device.type == pc.xyz.device.type, (
            f"Generator device type '{generator.device.type}' must match point cloud device type '{pc.xyz.device.type}'"
        )

        indices = torch.randperm(
            pc.xyz.shape[0], device=pc.xyz.device, generator=generator
        )
        return Select(indices)(pc)
