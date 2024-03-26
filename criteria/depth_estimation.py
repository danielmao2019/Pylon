import torch
from .base_criterion import BaseCriterion


class DepthEstimationLoss(BaseCriterion):

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
        assert y_true.min() >= 0, f"{y_true.min()=}"
        mask = y_true != 0
        assert mask.sum() >= 1
        loss = torch.nn.functional.l1_loss(y_pred[mask], y_true[mask], reduction='mean')
        assert loss.numel() == 1, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
