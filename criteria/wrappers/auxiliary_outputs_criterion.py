from typing import Sequence, Dict, Union
import torch
from ..base_criterion import BaseCriterion
from utils.builder import build_from_config


class AuxiliaryOutputsCriterion(BaseCriterion):
    __doc__ = r"""Compare multiple predictions to the same ground truth and sum all losses.
    """

    def __init__(self, cfg: dict) -> None:
        super(AuxiliaryOutputsCriterion, self).__init__()
        self.criterion = build_from_config(config=cfg)

    def __call__(
        self,
        y_pred: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        # input checks
        if type(y_pred) == dict:
            y_pred = list(y_pred.values())
        assert type(y_pred) in [tuple, list], f"{type(y_pred)=}"
        assert all(type(elem) == torch.Tensor for elem in y_pred)
        # compute losses
        loss: torch.Tensor = sum([
            self.criterion(y_pred=each_y_pred, y_true=y_true) for each_y_pred in y_pred
        ])
        assert loss.ndim == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
