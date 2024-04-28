from typing import Tuple, Dict, Optional
import torch
from .base_criterion import BaseCriterion


class SemanticSegmentationCriterion(BaseCriterion):

    def __init__(self, ignore_index: int, weight: Optional[Tuple[float, ...]] = None) -> None:
        super(SemanticSegmentationCriterion, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight, reduction='mean',
        )

    def __call__(self, y_pred: torch.Tensor, y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        y_true = y_true['mask']
        # input checks
        assert type(y_pred) == torch.Tensor, f"{type(y_pred)=}"
        assert y_pred.dim() == 4, f"{y_pred.shape=}"
        assert y_pred.is_floating_point(), f"{y_pred.dtype=}"
        assert type(y_true) == torch.Tensor, f"{type(y_true)=}"
        assert y_true.dim() == 3, f"{y_true.shape=}"
        assert y_true.dtype == torch.int64, f"{y_true.dtype=}"
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
        assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
        # compute loss
        loss = self.criterion(input=y_pred, target=y_true)
        assert loss.numel() == 1, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
