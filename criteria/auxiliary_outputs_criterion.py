from typing import List, Dict
import torch
from .base_criterion import BaseCriterion
from utils.builder import build_from_config


class AuxiliaryOutputsCriterion(BaseCriterion):

    def __init__(self, criterion_config: dict) -> None:
        super(AuxiliaryOutputsCriterion, self).__init__()
        self.criterion = build_from_config(config=criterion_config)

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        # input checks
        assert type(y_pred) == type(y_true) == dict, f"{type(y_pred)=}, {type(y_true)=}"
        # compute losses
        losses: List[torch.Tensor] = []
        for each_y_pred in y_pred.values():
            losses.append(self.criterion(y_pred=each_y_pred, y_true=y_true))
        # reduce
        loss: torch.Tensor = torch.stack(losses).sum(dim=0)
        assert loss.dim() == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
