from typing import Dict, Any
import torch
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud
from utils.point_cloud_ops.select import Select


class Shuffle(BaseTransform):

    def _call_single(self, pc: Dict[str, Any], generator: torch.Generator) -> Dict[str, Any]:
        check_point_cloud(pc)
        indices = torch.randperm(pc['pos'].shape[0], device=pc['pos'].device, generator=generator)
        return Select(indices)(pc)
