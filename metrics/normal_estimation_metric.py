from typing import Dict, Union
import torch
from .base_metric import BaseMetric
from utils.input_checks import check_normal_estimation


class NormalEstimationMetric(BaseMetric):

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, 3, H, W) for the (unnormalized) predicted normals.
            y_true (torch.Tensor or Dict[str, torch.Tensor]): a float32 tensor of shape (N, 3, H, W) for the (unnormalized) ground truth normals.

        Returns:
            score (torch.Tensor): a single-element tensor representing the score for normal estimation.
        """
        # input checks
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        check_normal_estimation(y_pred=y_pred, y_true=y_true)
        # compute score
        valid_mask = torch.linalg.norm(y_true, dim=1) != 0
        cosine_map = torch.nn.functional.cosine_similarity(y_pred, y_true, dim=1)
        assert valid_mask.shape == cosine_map.shape
        cosine_map = cosine_map.masked_select(valid_mask)
        score = torch.rad2deg(torch.acos(cosine_map)).mean()
        assert score.dim() == 0, f"{score.shape=}"
        # log score
        self.buffer.append(score)
        return score
