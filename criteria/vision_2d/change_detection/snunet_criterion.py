from typing import Dict
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d import SemanticSegmentationCriterion, DiceLoss


class SNUNetCDCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(SNUNetCDCriterion, self).__init__()
        self.semantic_criterion = SemanticSegmentationCriterion()
        self.dice_criterion = DiceLoss()

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_true.keys()) == {'change_map'}
        semantic_loss = self.semantic_criterion(y_pred=y_pred, y_true=y_true['change_map'])
        dice_loss = self.dice_criterion(y_pred=y_pred, y_true=y_true['change_map'])
        total_loss = dice_loss + semantic_loss
        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
