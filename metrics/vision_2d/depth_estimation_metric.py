import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_depth_estimation


class DepthEstimationMetric(SingleTaskMetric):

    DIRECTION = -1

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
        check_depth_estimation(y_pred=y_pred, y_true=y_true)
        y_pred = y_pred.squeeze(1)
        # compute score
        mask = y_true != 0
        assert mask.sum() >= 1
        numerator = ((y_pred[mask] - y_true[mask]) ** 2).sum()
        denominator = mask.sum()
        score = numerator / denominator
        assert score.ndim == 0, f"{score.shape=}"
        return {'l1': score}
