from typing import Dict, Any
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud


class UniformPosNoise(BaseTransform):

    def __init__(self, min: float = -0.1, max: float = 0.1) -> None:
        self.min = min
        self.max = max
        self.generator = torch.Generator()

    def _call_single_(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        check_point_cloud(pc)
        pc['pos'] += torch.rand(pc['pos'].shape, device=pc['pos'].device, generator=self.generator) * (self.max - self.min) + self.min
        return pc
