import torch
from .base_criterion import BaseCriterion


class NormalEstimationCriterion(BaseCriterion):

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): the (unnormalized) output of the model.
            y_true (torch.Tensor): the (unnormalized) ground truth for normal estimation.

        Returns:
            loss (torch.Tensor): a single-element tensor representing the loss for normal estimation.
        """
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert len(y_pred.shape) == len(y_true.shape) == 4, f"{y_pred.shape=}, {y_true.shape=}"
        # compute loss
        valid_mask = torch.linalg.norm(y_true, dim=1) != 0
        cosine_map = torch.nn.functional.cosine_similarity(y_pred, y_true, dim=1)
        assert valid_mask.shape == cosine_map.shape
        cosine_map = cosine_map.masked_select(valid_mask)
        loss = -cosine_map.mean()
        assert loss.numel() == 1, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
