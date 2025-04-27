from typing import Callable
import torch
from criteria.wrappers import SingleTaskCriterion


class PyTorchCriterionWrapper(SingleTaskCriterion):

    def __init__(self, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        super(PyTorchCriterionWrapper, self).__init__()
        assert callable(criterion), f"{type(criterion)=}"
        self.register_module('criterion', criterion)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(input=y_pred, target=y_true)
