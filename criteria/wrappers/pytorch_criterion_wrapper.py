from typing import Callable
import torch
from criteria.wrappers.single_task_criterion import SingleTaskCriterion


class PyTorchCriterionWrapper(SingleTaskCriterion):

    def __init__(
        self,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> None:
        super(PyTorchCriterionWrapper, self).__init__(**kwargs)
        assert callable(criterion), f"{type(criterion)=}"
        self.register_module('criterion', criterion)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(input=y_pred, target=y_true)
