import torch
from metrics.base_metric import BaseMetric


class InstanceSegmentationMetric(BaseMetric):

    def __init__(self, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index

    def _compute_score_(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # input checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute score
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        numerator = (y_pred[mask] - y_true[mask]).abs().sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.dim() == 0, f"{score.shape=}"
        return score
