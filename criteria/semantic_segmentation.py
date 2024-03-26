import torch
from .base_criterion import BaseCriterion


class SemanticSegmentationLoss(BaseCriterion):

    def __init__(self, num_classes: int, ignore_index: int):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index= ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # input checks
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
        # compute loss
        loss = self.criterion(input=y_pred, target=y_true)
        assert loss.numel() == 1, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
