import torch


class InstanceSegmentationMetric:

    def __init__(self, ignore_index: int):
        self.ignore_index = ignore_index

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        numerator = (y_pred[mask] - y_true[mask]).abs().sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.numel() == 1, f"{score.numel()}"
        return score
