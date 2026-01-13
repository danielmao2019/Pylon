from typing import Dict, List, Union
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud


class Select:

    def __init__(self, indices: Union[torch.Tensor, List[int]]) -> None:
        self.indices = indices

    def _materialize_indices(self, device: torch.device) -> torch.Tensor:
        if isinstance(self.indices, list):
            indices_tensor = torch.tensor(self.indices, dtype=torch.int64, device=device)
        else:
            assert self.indices.dtype == torch.int64, f"{self.indices.dtype=}"
            assert self.indices.device == device, f"{self.indices.device=}, {device=}"
            indices_tensor = self.indices
        assert torch.all(indices_tensor >= 0), f"Negative indices not allowed: {indices_tensor}"
        return indices_tensor

    def __call__(self, pc: PointCloud) -> PointCloud:
        assert isinstance(pc, PointCloud), f"{type(pc)=}"
        indices = self._materialize_indices(device=pc.xyz.device)
        assert torch.all(indices < pc.num_points), f"indices exceed length {pc.num_points}: {indices=}"

        data: Dict[str, torch.Tensor] = {'xyz': pc.xyz[indices]}
        if hasattr(pc, 'indices'):
            data['indices'] = pc.indices[indices]
        else:
            data['indices'] = indices
        for name in pc.field_names()[1:]:
            if name == 'indices':
                continue
            data[name] = getattr(pc, name)[indices]
        return PointCloud(data=data)

    def __str__(self) -> str:
        if isinstance(self.indices, list):
            num_indices = len(self.indices)
            if num_indices <= 5:
                return f"Select(indices={self.indices})"
            return f"Select(indices=[...{num_indices} indices])"
        num_indices = self.indices.numel()
        if num_indices <= 5:
            return f"Select(indices={self.indices.tolist()})"
        return f"Select(indices=[...{num_indices} indices])"
