from typing import Dict
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d import DiceLoss


class LWGANetCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(LWGANetCriterion, self).__init__()
        # unbounded
        self.bce_criterion = torch.nn.functional.binary_cross_entropy_with_logits()
        self.dice_criterion = DiceLoss()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_true.keys()) == {'change_map'}
        bce_loss += (self.bce_criterion(y_pred=mask, y_true=y_true['change_map']) for mask in y_pred.values()) 
        dice_loss += (self.dice_criterion(y_pred=mask, y_true=y_true['change_map']) for mask in y_pred.values()) 
        
        return bce_loss + dice_loss
