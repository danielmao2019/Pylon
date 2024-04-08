from typing import Tuple, List
import random
import torch
from .diffusion_dataset_wrapper import DiffusionDatasetWrapper
from utils.object_detection import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy


class BBoxDiffusionDatasetWrapper(DiffusionDatasetWrapper):

    def __init__(
        self,
        dataset: dict,
        num_steps: int,
        keys: List[Tuple[str, str]],
        num_bboxes: int,
        scale: float,
    ):
        super(BBoxDiffusionDatasetWrapper, self).__init__(dataset=dataset, num_steps=num_steps, keys=keys)
        assert type(num_bboxes) == int, f"{type(num_bboxes)=}"
        self.num_bboxes = num_bboxes
        assert type(scale) == float, f"{type(scale)=}"
        self.scale = scale

    def forward_diffusion(self, bboxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward diffusion for bounding box annotations in a single scene.
        This method overrides and calls `DiffusionDatasetWrapper.forward_diffusion`.

        Args:
            bboxes (torch.Tensor): float32 tensor of shape (N, 4).
                Assuming in format (x0, y0, x1, y1).

        Returns:
            torch.Tensor: noisy bounding boxes.
            torch.Tensor: time step.
        """
        assert type(bboxes) == torch.Tensor, f"{type(bboxes)=}"
        assert len(bboxes.shape) == 2 and bboxes.shape[1] == 4, f"{bboxes.shape=}"
        bboxes = bbox_xyxy_to_cxcywh(bboxes)
        if len(bboxes) < self.num_bboxes:
            padding = torch.randn(size=(self.num_bboxes-len(bboxes), 4)) / 6. + 0.5
            padding[:, 2:] = torch.clamp(padding[:, 2:], min=1e-4)
            bboxes = torch.cat([bboxes, padding], dim=0)
        elif len(bboxes) > self.num_bboxes:
            mask = [True] * self.num_bboxes + [False] * (len(bboxes)-self.num_bboxes)
            random.shuffle(mask)
            bboxes = bboxes[mask]
        else:
            pass
        # affine transform from range [0, 1] to [-scale, +scale]
        bboxes = (bboxes * 2 - 1) * self.scale
        bboxes, time = super(BBoxDiffusionDatasetWrapper, self).forward_diffusion(bboxes)
        bboxes = torch.clamp(bboxes, min=-self.scale, max=+self.scale)
        # affine transform from range [-scale, +scale] to [0, 1]
        bboxes = ((bboxes / self.scale) + 1) / 2
        assert torch.all(bboxes >= 0)
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        bboxes = torch.clamp(bboxes, min=0, max=1)
        return bboxes, time
