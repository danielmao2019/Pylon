import torch
from .base_metric import BaseMetric


class NormalEstimationMetric(BaseMetric):

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): the (unnormalized) output of the model.
            y_true (torch.Tensor): the (unnormalized) ground truth for normal estimation.

        Returns:
            score (torch.Tensor): a single-element tensor representing the score for normal estimation.
        """
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert len(y_pred.shape) == len(y_true.shape) == 4, f"{y_pred.shape=}, {y_true.shape=}"
        # compute score
        valid_mask = torch.linalg.norm(y_true, dim=1) != 0
        cosine_map = torch.nn.functional.cosine_similarity(y_pred, y_true, dim=1)
        assert valid_mask.shape == cosine_map.shape
        cosine_map = cosine_map.masked_select(valid_mask)
        score = torch.rad2deg(torch.acos(cosine_map)).mean()
        assert score.numel() == 1, f"{score.numel()=}"
        # log score
        self.buffer.append(score)
        return score
