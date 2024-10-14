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
        mask = y_true != self.ignore_index
        assert mask.sum() >= 1
        numerator = (y_pred[mask] - y_true[mask]).abs().sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.ndim == 0, f"{score.shape=}"
        return {'l1': score}
