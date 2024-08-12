from typing import Callable
import torch
from ..base_criterion import BaseCriterion


class PyTorchCriterionWrapper(BaseCriterion):

    def __init__(self, criterion: Callable) -> None:
        super(PyTorchCriterionWrapper, self).__init__()
        assert callable(criterion), f"{type(criterion)=}"
        self.criterion = criterion

    def _compute_loss_(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(input=y_pred, target=y_true)
