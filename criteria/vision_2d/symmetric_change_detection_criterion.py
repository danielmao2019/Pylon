from typing import Dict
from criteria.vision_2d import SemanticSegmentationCriterion
import torch


class SymmetricChangeDetectionCriterion(SemanticSegmentationCriterion):

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        assert type(y_pred) == dict and set(y_pred.keys()) == {'change_map_12', 'change_map_21'}
        assert type(y_true) == torch.Tensor
        loss_12 = self.criterion(y_pred['change_map_12'], y_true)
        loss_21 = self.criterion(y_pred['change_map_21'], y_true)
        total_loss = (loss_12 + loss_21) / 2
        return total_loss
