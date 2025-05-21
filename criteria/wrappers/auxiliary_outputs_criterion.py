from typing import Sequence, Dict, Union, Optional, Callable, Any
import torch
from criteria.base_criterion import BaseCriterion
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from utils.builders import build_from_config


class AuxiliaryOutputsCriterion(SingleTaskCriterion):
    __doc__ = r"""Compare multiple predictions to the same ground truth and sum all losses.
    """

    REDUCTION_OPTIONS = ['sum', 'mean']

    def __init__(
        self,
        criterion_cfg: Dict[str, Union[Callable, Dict[str, Any]]],
        reduction: Optional[str] = 'sum',
        **kwargs,
    ) -> None:
        super(AuxiliaryOutputsCriterion, self).__init__(**kwargs)
        assert reduction in self.REDUCTION_OPTIONS
        self.reduction = reduction
        # Build criterion as submodule
        assert isinstance(criterion_cfg, dict) and criterion_cfg.keys() == {'class', 'args'}, \
            f"{type(criterion_cfg)=}, {criterion_cfg.keys()=}"
        criterion_cfg['args']['use_buffer'] = False  # Disable buffer for component criterion
        criterion = build_from_config(config=criterion_cfg)
        self.register_module('criterion', criterion)
        assert isinstance(self.criterion, BaseCriterion)

    def __call__(
        self,
        y_pred: Union[Sequence[torch.Tensor], Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        # input checks
        if type(y_pred) == dict:
            y_pred = list(y_pred.values())
        assert isinstance(y_pred, (tuple, list)), f"{type(y_pred)=}"
        assert all(isinstance(elem, torch.Tensor) for elem in y_pred), \
            f"{[type(elem) for elem in y_pred]}"
        # compute losses
        losses: torch.Tensor = torch.stack([
            self.criterion(y_pred=each_y_pred, y_true=y_true) for each_y_pred in y_pred
        ], dim=0)
        if self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        self.add_to_buffer(loss)
        return loss
