from typing import Dict
import torch
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion


class ChangeStarCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs) -> None:
        super(ChangeStarCriterion, self).__init__(**kwargs)
        self.criterion = SemanticSegmentationCriterion()

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
        self.add_to_buffer(total_loss)
        return total_loss
