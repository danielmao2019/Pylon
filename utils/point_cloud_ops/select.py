from typing import Any, Dict, Union, List
import torch
from utils.input_checks.check_point_cloud import check_point_cloud


class Select:

    def __init__(self, indices: Union[torch.Tensor, List[int]]) -> None:
        self.indices = indices

    def __call__(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        check_point_cloud(pc)
        if isinstance(self.indices, list):
            self.indices = torch.tensor(
                self.indices, dtype=torch.int64, device=pc['pos'].device
            )
        else:
            assert self.indices.dtype == torch.int64
            assert (
                self.indices.device == pc['pos'].device
            ), f"{self.indices.device=}, {pc['pos'].device=}"

        # Validate indices are non-negative
        assert torch.all(
            self.indices >= 0
        ), f"Negative indices not allowed, got: {self.indices[self.indices < 0].tolist()}"

        result = {}
        for key, val in pc.items():
            if key == 'indices':
                continue
            result[key] = val[self.indices]
        result['indices'] = (
            pc['indices'][self.indices] if 'indices' in pc else self.indices
        )
        return result

    def __str__(self) -> str:
        """String representation of the Select transform."""
        if isinstance(self.indices, list):
            num_indices = len(self.indices)
            if num_indices <= 5:
                return f"Select(indices={self.indices})"
            else:
                return f"Select(indices=[...{num_indices} indices])"
        else:
            num_indices = self.indices.numel()
            if num_indices <= 5:
                return f"Select(indices={self.indices.tolist()})"
            else:
                return f"Select(indices=[...{num_indices} indices])"
