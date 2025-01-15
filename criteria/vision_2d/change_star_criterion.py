from typing import Dict
import torch
import criteria
import criteria
from criteria.wrappers import SingleTaskCriterion


class ChangeStarCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        self.criterion = criteria.vision_2d.SemanticSegmentationCriterion()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_pred.keys()) == set(['change_12', 'change_21', 'semantic'])
        assert set(y_true.keys()) == set(['change', 'semantic'])
        change_12_loss = self.criterion(y_pred=y_pred['change_12'], y_true=y_true['change'])
        change_21_loss = self.criterion(y_pred=y_pred['change_21'], y_true=y_true['change'])
        change_loss = 0.5 * (change_12_loss + change_21_loss)
        semantic_loss = self.criterion(y_pred=y_pred['semantic'], y_true=y_true['semantic'])
        total_loss = change_loss + semantic_loss
        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        return total_loss
