from typing import Any, Dict
import torch
from utils.point_cloud_ops.select import Select


class RandomSelect:
    def __init__(self, percentage: float) -> None:
        assert isinstance(percentage, (int, float)), f"{type(percentage)=}"
        assert 0 <= percentage <= 1, f"{percentage=}"
        self.percentage = percentage

    def __call__(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        assert isinstance(pc, dict), f"{type(pc)=}"
        assert pc.keys() >= {'pos'}, f"{pc.keys()=}"
        assert pc['pos'].ndim == 2 and pc['pos'].shape[1] == 3, f"{pc['pos'].shape=}"
        assert pc['pos'].dtype == torch.float32, f"{pc['pos'].dtype=}"

        num_points = pc['pos'].shape[0]
        num_points_to_select = int(num_points * self.percentage)
        indices = torch.randperm(num_points)[:num_points_to_select]
        return Select(indices)(pc)
