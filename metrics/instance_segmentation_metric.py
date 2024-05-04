from typing import Dict, Union
import torch
from .base_metric import BaseMetric


class InstanceSegmentationMetric(BaseMetric):

    def __init__(self, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        # input checks
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute score
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        numerator = (y_pred[mask] - y_true[mask]).abs().sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.dim() == 0, f"{score.shape=}"
        # log score
        self.buffer.append(score)
        return score
