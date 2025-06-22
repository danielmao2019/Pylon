from typing import Dict
import torch
from criteria.vision_2d import SemanticSegmentationCriterion, DiceLoss
from criteria.wrappers import SingleTaskCriterion


class SNUNetCDCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs) -> None:
        super(SNUNetCDCriterion, self).__init__(**kwargs)
        self.semantic_criterion = SemanticSegmentationCriterion()
        self.dice_criterion = DiceLoss()

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_true.keys()) == {'change_map'}
        semantic_loss = self.semantic_criterion(y_pred=y_pred, y_true=y_true['change_map'])
        dice_loss = self.dice_criterion(y_pred=y_pred, y_true=y_true['change_map'])
        total_loss = dice_loss + semantic_loss
        self.add_to_buffer(total_loss)
        return total_loss
