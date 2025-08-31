from typing import List
import torch


def multi_view_fusion(
    points: torch.Tensor,
    maps: List[torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
) -> torch.Tensor:
    pass
