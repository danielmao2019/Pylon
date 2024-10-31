from typing import Dict
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.input_checks import check_normal_estimation


NUMERICAL_STABILITY = 1.0e-05


class NormalEstimationMetric(SingleTaskMetric):

    DIRECTION = -1

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, 3, H, W) for the (unnormalized) predicted normals.
            y_true (torch.Tensor): a float32 tensor of shape (N, 3, H, W) for the (unnormalized) ground truth normals.

        Returns:
            score (Dict[str, torch.Tensor]): a dictionary with the following fields
            {
                'angle': a single-element tensor representing the angle
                    between pred and true normal vectors.
            }
        """
        # input checks
        check_normal_estimation(y_pred=y_pred, y_true=y_true)
        # compute cosine similarities
        cosine_map = torch.nn.functional.cosine_similarity(y_pred, y_true, dim=1)
        assert torch.all(torch.maximum(cosine_map - 1, -1 - cosine_map) < NUMERICAL_STABILITY), f"{cosine_map.min()=}, {cosine_map.max()=}"
        cosine_map = torch.clamp(cosine_map, min=-1, max=+1)
        # filter by valid mask
        valid_mask = torch.linalg.norm(y_true, dim=1) != 0
        assert valid_mask.sum() > 0
        assert valid_mask.shape == cosine_map.shape, f"{valid_mask.shape=}, {cosine_map.shape=}"
        cosine_map = cosine_map.masked_select(valid_mask)
        # compute score
        score = torch.rad2deg(torch.acos(cosine_map)).mean()
        # output check
        assert score.ndim == 0, f"{score.shape=}"
        assert score.is_floating_point(), f"{score.dtype=}"
        assert not torch.isnan(score), f"{score=}, {cosine_map.min()=}, {cosine_map.max()=}, {torch.any(torch.isnan(cosine_map))=}, {torch.any(torch.isnan(torch.acos(cosine_map)))=}, {torch.any(torch.isnan(torch.rad2deg(torch.acos(cosine_map))))=}"
        # return
        return {'angle': score}
