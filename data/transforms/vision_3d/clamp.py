from typing import Dict, Any
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud
from utils.point_cloud_ops.select import Select


class Clamp(BaseTransform):

    def __init__(self, max_count: int) -> None:
        assert isinstance(max_count, int)
        assert max_count > 0, f"{max_count=}"
        self.max_count = max_count
        super(Clamp, self).__init__()

    def _call_single_(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        check_point_cloud(pc)
        num_points = pc['pos'].shape[0]
        if num_points > self.max_count:
            indices = torch.arange(self.max_count, device=pc['pos'].device)
            return Select(indices)(pc)
        else:
            return pc
