from typing import Tuple
import torch


class ResizeBBoxes:

    def __init__(self, scale_factor: Tuple[float, float], target_size: Tuple[int, int]) -> None:
        r"""
        Args:
            scale_factor (Tuple[float, float]): The scale factor at which the bounding box coordinated will be resized.
            target_size (Tuple[int, int]): The size of the entire scene after resize. Used for clipping.
        """
        assert type(scale_factor) == tuple, f"{type(scale_factor)=}"
        assert len(scale_factor) == 2, f"{len(scale_factor)=}"
        self.scale_factor = scale_factor
        assert type(target_size) == tuple, f"{type(target_size)=}"
        assert len(target_size) == 2, f"{len(target_size)=}"
        self.target_size = target_size

    def __call__(self, boxes: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            boxes (torch.Tensor): Bounding box annotations. Assumed to be in format (x1, y1, x2, y2).
        """
        assert len(boxes.shape) == 2 and boxes.shape[1] == 4, f"{boxes.shape=}"
        boxes[:, 1] = torch.clip(boxes[:, 1]*self.scale_factor[0], min=0, max=self.target_size[0])
        boxes[:, 3] = torch.clip(boxes[:, 3]*self.scale_factor[0], min=0, max=self.target_size[0])
        boxes[:, 0] = torch.clip(boxes[:, 0]*self.scale_factor[1], min=0, max=self.target_size[1])
        boxes[:, 2] = torch.clip(boxes[:, 2]*self.scale_factor[1], min=0, max=self.target_size[1])
        return boxes
