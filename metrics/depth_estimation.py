import torch
from .base_metric import BaseMetric


class DepthEstimationMetric(BaseMetric):

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert y_true.min() >= 0, f"{y_true.min()=}"
        mask = y_true != 0
        assert mask.sum() >= 1
        numerator = ((y_pred[mask] - y_true[mask]) ** 2).sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.numel() == 1, f"{score.numel()=}"
        # log score
        self.buffer.append(score)
        return score
