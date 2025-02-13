from typing import List, Dict
import torch
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d import SemanticSegmentationCriterion, IoULoss, SSIMLoss


class FTNCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        self.ce_loss = SemanticSegmentationCriterion()
        self.ssim_loss = SSIMLoss()
        self.iou_loss = IoULoss()

    def __call__(self, y_pred: List[torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        # input checks
        assert isinstance(y_pred, list) and len(y_pred) == 4
        assert all(isinstance(x, torch.Tensor) for x in y_pred)
        assert isinstance(y_true, dict) and set(y_true.keys()) == {'change_map'}
        # compute losses
        ce_losses = torch.stack([
            self.ce_loss(x, torch.nn.functional.interpolate(y_true['change_map'], size=x.shape[-2:], mode='nearest'))
            for x in y_pred
        ], dim=0)
        ssim_losses = torch.stack([
            self.ssim_loss(torch.nn.Softmax(dim=1)(x), torch.nn.functional.interpolate(y_true['change_map'], size=x.shape[-2:], mode='nearest'))
            for x in y_pred
        ], dim=0)
        iou_losses = torch.stack([
            self.iou_loss(torch.nn.Softmax(dim=1)(x), torch.nn.functional.interpolate(y_true['change_map'], size=x.shape[-2:], mode='nearest'))
            for x in y_pred
        ], dim=0)
        total_loss = torch.sum(ce_losses) + torch.sum(ssim_losses) + torch.sum(iou_losses)
        assert total_loss.ndim == 0, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
