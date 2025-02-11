from typing import Dict
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion
from dice_loss import DiceLoss


class SNUNet_CD_Criterion(SingleTaskCriterion):

    def __init__(self) -> None:
        super(SNUNet_CD_Criterion, self).__init__()
        self.semantic_criterion = lambda y_pred, y_true :criteria.vision_2d.SemanticSegmentationCriterion(y_pred, y_true)
        self.dice_criterion = lambda y_pred, y_true: criteria.vision_2d.DiceLoss(y_pred, y_true)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override parent class __call__ method.
        """
        assert set(y_pred.keys()) == set(['change_map', 'semantic'])
        assert set(y_true.keys()) == set(['change_map', 'semantic'])
        semantic_loss = self.semantic_criterion(y_pred=y_pred['semantic'], y_true=y_true['semantic'])
        dice_loss = self.dice_criterion(y_pred=y_pred['change_map'], y_true=y_true['change_map'])
        total_loss = dice_loss + semantic_loss
        assert total_loss.numel() == 1, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
