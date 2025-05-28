from typing import Any, Dict, Union, List
import torch
from utils.input_checks.point_cloud import check_point_cloud


class Select:

    def __init__(self, indices: Union[torch.Tensor, List[int]]) -> None:
        self.indices = indices

    def __call__(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        check_point_cloud(pc)
        if isinstance(self.indices, list):
            self.indices = torch.tensor(self.indices, dtype=torch.int64, device=pc['pos'].device)
        else:
            assert self.indices.device == pc['pos'].device, f"{self.indices.device=}, {pc['pos'].device=}"
        result = {}
        for key, val in pc.items():
            if key == 'indices':
                continue
            result[key] = val[self.indices]
        result['indices'] = result['indices'][self.indices] if 'indices' in result else self.indices
        return result
