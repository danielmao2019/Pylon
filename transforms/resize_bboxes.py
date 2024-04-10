from typing import Tuple
import torch


class ResizeBBoxes:

    def __init__(self, scale_factor: Tuple[float, float]) -> None:
        r"""
        Args:
            scale_factor (Tuple[float, float]): The scale factor at which the bounding box coordinated will be resized.
                Assumed in format (height, width).
        """
        assert type(scale_factor) == tuple, f"{type(scale_factor)=}"
        assert len(scale_factor) == 2, f"{len(scale_factor)=}"
        assert type(scale_factor[0]) == type(scale_factor[1]) == float, f"{type(scale_factor[0])=}, {type(scale_factor[1])=}"
        self.scale_factor = scale_factor

    def __call__(self, bboxes: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            boxes (torch.Tensor): Bounding box annotations. Assumed in format (x1, y1, x2, y2).
        """
        assert type(bboxes) == torch.Tensor, f"{type(bboxes)=}"
        assert bboxes.dim() == 2 and bboxes.shape[1] == 4, f"{bboxes.shape=}"
        assert torch.is_floating_point(bboxes), f"{bboxes.dtype=}"
        assert 0 <= bboxes.min() <= bboxes.max() <= 1, f"{bboxes.min()=}, {bboxes.max()=}"
        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2]*self.scale_factor[1], min=0, max=1)
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2]*self.scale_factor[0], min=0, max=1)
        return bboxes
