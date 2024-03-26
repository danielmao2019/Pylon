import torch
from .base_criterion import BaseCriterion


class InstanceSegmentationLoss(BaseCriterion):

    def __init__(self, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        loss = torch.nn.functional.l1_loss(y_pred[mask], y_true[mask], reduction='mean')
        assert loss.numel() == 1, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
