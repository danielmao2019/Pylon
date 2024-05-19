import torch
from .base_criterion import BaseCriterion


class InstanceSegmentationCriterion(BaseCriterion):

    def __init__(self, ignore_index: int) -> None:
        super(InstanceSegmentationCriterion, self).__init__()
        self.ignore_index = ignore_index

    def _compute_loss_(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # input checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute loss
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        return torch.nn.functional.l1_loss(y_pred[mask], y_true[mask], reduction='mean')
