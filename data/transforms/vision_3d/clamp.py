from typing import Dict, Any
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud
from utils.point_cloud_ops.random_select import RandomSelect


class Clamp(BaseTransform):

    def __init__(self, max_points: int) -> None:
        assert isinstance(max_points, int)
        assert max_points > 0, f"{max_points=}"
        self.max_points = max_points
        super(Clamp, self).__init__()

    def _call_single(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            pc (Dict[str, Any]): The point cloud to clamp.
        """
        check_point_cloud(pc)
        num_points = pc['pos'].shape[0]
        if num_points > self.max_points:
            return RandomSelect(count=self.max_points)(pc)
        else:
            return pc
