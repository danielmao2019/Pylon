from typing import Sequence, Dict, Union, Optional
import torch
from .single_task_criterion import SingleTaskCriterion
from utils.builders import build_from_config


class AuxiliaryOutputsCriterion(SingleTaskCriterion):
    __doc__ = r"""Compare multiple predictions to the same ground truth and sum all losses.
    """

    REDUCTION_OPTIONS = ['sum', 'mean']

    def __init__(self, criterion_cfg: dict, reduction: Optional[str] = 'sum') -> None:
        super(AuxiliaryOutputsCriterion, self).__init__()
        self.criterion = build_from_config(config=criterion_cfg)
        assert reduction in self.REDUCTION_OPTIONS
        self.reduction = reduction

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
        losses: torch.Tensor = torch.stack([
            self.criterion(y_pred=each_y_pred, y_true=y_true) for each_y_pred in y_pred
        ], dim=0)
        if self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        assert loss.ndim == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss.detach().cpu())
        return loss
