from typing import Dict
import torch
from criteria.vision_2d import SemanticSegmentationCriterion


class SymmetricChangeDetectionCriterion(SemanticSegmentationCriterion):

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert type(y_pred) == dict and set(y_pred.keys()) == {'change_map_12', 'change_map_21'}
        assert type(y_true) == dict and set(y_true.keys()) == {'change_map'}
        loss_12 = self.criterion(y_pred['change_map_12'], y_true['change_map'])
        loss_21 = self.criterion(y_pred['change_map_21'], y_true['change_map'])
        total_loss = (loss_12 + loss_21) / 2
        self.add_to_buffer(total_loss)
        return total_loss
