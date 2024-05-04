from typing import Tuple, List, Dict, Union
import torch
from .base_criterion import BaseCriterion
from utils.builder import build_from_config


class AuxiliaryOutputsCriterion(BaseCriterion):
    __doc__ = r"""Compare multiple predictions to the same ground truth and sum all losses.
    """

    def __init__(self, criterion_config: dict) -> None:
        super(AuxiliaryOutputsCriterion, self).__init__()
        self.criterion = build_from_config(config=criterion_config)

    def __call__(
        self,
        y_pred: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        # input checks
        if type(y_pred) == dict:
            y_pred = list(y_pred.values())
        else:
            assert type(y_pred) in [tuple, list]
        # compute losses
        loss: torch.Tensor = sum([
            self.criterion(y_pred=each_y_pred, y_true=y_true) for each_y_pred in y_pred
        ])
        assert loss.dim() == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
