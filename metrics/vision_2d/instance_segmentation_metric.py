from typing import Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


class InstanceSegmentationMetric(SingleTaskMetric):

    DIRECTION = -1

    def __init__(self, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Args:
            y_pred (torch.Tensor)
            y_true (torch.Tensor)

        Returns:
            score (Dict[str, torch.Tensor]): a dictionary with the following fields
            {
                'l1': a single-element tensor representing the l1 distance between pred and true.
            }
        """
        # input checks
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        # compute score
        valid_mask = y_true != self.ignore_index
        assert valid_mask.sum() > 0
        numerator = (y_pred[valid_mask] - y_true[valid_mask]).abs().sum()
        denominator = valid_mask.sum()
        score = numerator / denominator
        # output check
        assert score.ndim == 0, f"{score.shape=}"
        assert score.is_floating_point(), f"{score.dtype=}"
        assert not torch.isnan(score), f"{score=}"
        return {'l1': score}
