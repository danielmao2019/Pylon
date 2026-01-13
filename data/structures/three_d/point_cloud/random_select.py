from typing import Any, Optional, Union
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select


class RandomSelect:

    def __init__(
        self, percentage: Optional[float] = None, count: Optional[int] = None
    ) -> None:
        assert (percentage is not None) ^ (count is not None), f"{percentage=}, {count=}"
        if percentage is not None:
            assert isinstance(percentage, (int, float)), f"{type(percentage)=}"
            assert 0 < percentage <= 1, f"{percentage=}"
            self.percentage = float(percentage)
            self.count = None
        else:
            assert isinstance(count, int), f"{type(count)=}"
            assert count > 0, f"{count=}"
            self.count = count
            self.percentage = None

    def __call__(
        self,
        pc: PointCloud,
        seed: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
    ) -> PointCloud:
        assert (seed is not None) ^ (generator is not None), f"{seed=}, {generator=}"

        assert isinstance(pc, PointCloud), f"{type(pc)=}"
        device = pc.xyz.device
        num_points = pc.num_points

        if generator is not None:
            assert generator.device.type == device.type, f"{generator.device.type=}, {device.type=}"
            gen = generator
        else:
            gen = torch.Generator(device=device)
            if not isinstance(seed, int):
                from utils.determinism.hash_utils import convert_to_seed

                seed = convert_to_seed(seed)
            gen.manual_seed(seed)

        if self.percentage is not None:
            num_points_to_select = int(num_points * self.percentage)
        else:
            num_points_to_select = min(self.count, num_points)

        indices = torch.randperm(num_points, generator=gen, device=device)[
            :num_points_to_select
        ]
        return Select(indices=indices)(pc)

    def __str__(self) -> str:
        if self.percentage is not None:
            return f"RandomSelect(percentage={self.percentage})"
        return f"RandomSelect(count={self.count})"
