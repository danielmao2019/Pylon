from typing import Any, Dict
import torch
from utils.input_checks.point_cloud import check_point_cloud


class Select:
    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices

    def __call__(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        check_point_cloud(pc)
        result = {}
        for key, val in pc.items():
            result[key] = val[self.indices]
        result['indices'] = self.indices
        return result
