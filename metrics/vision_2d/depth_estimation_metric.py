import torch
from metrics.base_metric import BaseMetric
from utils.input_checks import check_depth_estimation


class DepthEstimationMetric(BaseMetric):

    def _compute_score_(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # input checks
        check_depth_estimation(y_pred=y_pred, y_true=y_true)
        y_pred = y_pred.squeeze(1)
        # compute score
        mask = y_true != 0
        assert mask.sum() >= 1
        numerator = ((y_pred[mask] - y_true[mask]) ** 2).sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.ndim == 0, f"{score.shape=}"
        return score
