from typing import Dict, Union
import torch
from .base_metric import BaseMetric
from utils.input_checks import check_depth_estimation


class DepthEstimationMetric(BaseMetric):

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        # input checks
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        check_depth_estimation(y_pred=y_pred, y_true=y_true)
        y_pred = y_pred.squeeze(1)
        # compute score
        mask = y_true != 0
        assert mask.sum() >= 1
        numerator = ((y_pred[mask] - y_true[mask]) ** 2).sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.dim() == 0, f"{score.shape=}"
        # log score
        self.buffer.append(score)
        return score
