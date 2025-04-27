from typing import Tuple, Dict, Optional
import torch
from criteria.wrappers import SingleTaskCriterion, AuxiliaryOutputsCriterion
from criteria.vision_2d import SemanticSegmentationCriterion, IoULoss, SSIMLoss


class FTNCriterion(SingleTaskCriterion):

    def __init__(self, num_classes: Optional[int] = 2) -> None:
        super(FTNCriterion, self).__init__()
        self.ce_criterion = AuxiliaryOutputsCriterion(SemanticSegmentationCriterion())
        self.ssim_criterion = AuxiliaryOutputsCriterion(SSIMLoss(channels=num_classes))
        self.iou_criterion = AuxiliaryOutputsCriterion(IoULoss())

    def __call__(self, y_pred: Tuple[torch.Tensor, ...], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        # input checks
        assert isinstance(y_pred, tuple) and len(y_pred) == 4, f"{type(y_pred)=}, {len(y_pred)=}"
        assert all(isinstance(x, torch.Tensor) for x in y_pred)
        assert isinstance(y_true, dict) and set(y_true.keys()) == {'change_map'}
        # compute losses
        ce_loss = self.ce_criterion(y_pred, y_true['change_map'])
        ssim_loss = self.ssim_criterion(y_pred, y_true['change_map'])
        iou_loss = self.iou_criterion(y_pred, y_true['change_map'])
        total_loss = ce_loss + ssim_loss + iou_loss
        self.add_to_buffer(total_loss)
        return total_loss
