from typing import Dict, Any
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud


class UniformPosNoise(BaseTransform):

    def __init__(self, min: float = -0.1, max: float = 0.1) -> None:
        self.min = min
        self.max = max

    def _call_single(self, pc: Dict[str, Any], generator: torch.Generator) -> Dict[str, Any]:
        check_point_cloud(pc)
        
        # Validate generator device type matches point cloud device type
        assert generator.device.type == pc['pos'].device.type, (
            f"Generator device type '{generator.device.type}' must match point cloud device type '{pc['pos'].device.type}'"
        )
        
        pc['pos'] += torch.rand(pc['pos'].shape, device=pc['pos'].device, generator=generator) * (self.max - self.min) + self.min
        return pc
