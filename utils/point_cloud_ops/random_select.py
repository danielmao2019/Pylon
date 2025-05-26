from typing import Any, Dict
import torch
from utils.input_checks import check_point_cloud
from utils.point_cloud_ops.select import Select


class RandomSelect:
    def __init__(self, percentage: float) -> None:
        assert isinstance(percentage, (int, float)), f"{type(percentage)=}"
        assert 0 <= percentage <= 1, f"{percentage=}"
        self.percentage = percentage

    def __call__(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        check_point_cloud(pc)
        num_points = pc['pos'].shape[0]
        num_points_to_select = int(num_points * self.percentage)
        indices = torch.randperm(num_points)[:num_points_to_select]
        return Select(indices)(pc)
