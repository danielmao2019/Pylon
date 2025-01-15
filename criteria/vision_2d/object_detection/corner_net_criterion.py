from typing import List, Dict
import criteria.common
import torch
import criteria
from criteria.wrappers import SingleTaskCriterion


class CornerNerCriterion(SingleTaskCriterion):

    def __init__(self) -> None:
        self.focal_loss_criterion = criteria.common

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
        # Input checks
        assert isinstance(y_pred, dict) and set(y_pred.keys()) == set(['labels', 'bboxes'])
        assert isinstance(y_pred['labels'], torch.Tensor) and y_pred['labels'].ndim == 2
        assert isinstance(y_pred['bboxes'], torch.Tensor) and y_pred['bboxes'].ndim == 3
        assert y_pred['bboxes'].shape == y_pred['labels'].shape + (4,)
        assert isinstance(y_true, dict) and set(y_true.keys()) == set(['labels', 'bboxes'])
        assert isinstance(y_true['labels'], list) and all(isinstance(x, torch.Tensor) for x in y_true['labels'])
        assert isinstance(y_true['bboxes'], list) and all(isinstance(x, torch.Tensor) for x in y_true['bboxes'])
        assert all(b.shape == l.shape + (4,) for b, l in zip(y_true['labels'], y_true['bboxes']))
        # Compute focal loss
        focal_loss: torch.Tensor = 
