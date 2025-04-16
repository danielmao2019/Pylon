from typing import Any, Dict
import torch


class Select:
    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices

    def __call__(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        assert isinstance(pc, dict), f"{type(pc)=}"
        assert pc.keys() >= {'pos'}, f"{pc.keys()=}"
        assert pc['pos'].ndim == 2 and pc['pos'].shape[1] == 3, f"{pc['pos'].shape=}"
        assert pc['pos'].dtype == torch.float32, f"{pc['pos'].dtype=}"
        result = {}
        for key, val in pc.items():
            if isinstance(val, torch.Tensor) and val.size(0) == pc['pos'].size(0):
                result[key] = val[self.indices]
        result['indices'] = self.indices
        return result
