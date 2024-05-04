from typing import Dict, Union
import torch
from .base_criterion import BaseCriterion


class InstanceSegmentationCriterion(BaseCriterion):

    def __init__(self, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""
        Returns:
            loss (torch.Tensor): a float32 scalar tensor for loss value.
        """
        # input checks
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute loss
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        loss = torch.nn.functional.l1_loss(y_pred[mask], y_true[mask], reduction='mean')
        assert loss.dim() == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
