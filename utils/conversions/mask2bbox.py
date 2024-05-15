from typing import Tuple
import torch


def mask2bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    r"""Draws the tightest bounding box around the region with `True` values.
    """
    assert type(mask) == torch.Tensor, f"{type(mask)=}"
    assert mask.dim() == 2, f"{mask.shape=}"
    assert mask.dtype == torch.bool, f"{mask.dtype=}"
    region = torch.nonzero(mask)
    y1 = region[:, 0].min().item()
    y2 = region[:, 0].max().item()
    x1 = region[:, 1].min().item()
    x2 = region[:, 1].max().item()
    return (x1, y1, x2, y2)
