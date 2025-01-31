from typing import Dict
import torch
from semantic_segmentation_criterion import SemanticSegmentationCriterion
from dice_loss import DiceLoss


class SNUNetCd(SingleTaskCriterion):

    def __init__(self) -> None:
        super(SNUNetCd, self).__init__()
        self.criterion = lambda y_pred, y_true :SemanticSegmentationCriterion(y_pred, y_true) + DiceLoss(y_pred, y_true)

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
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
