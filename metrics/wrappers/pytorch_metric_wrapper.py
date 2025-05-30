from typing import Callable
import torch
from metrics.wrappers.single_task_metric import SingleTaskMetric


class PyTorchMetricWrapper(SingleTaskMetric):

    def __init__(
        self,
        metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        **kwargs,
    ) -> None:
        super(PyTorchMetricWrapper, self).__init__(**kwargs)
        assert callable(metric), f"{type(metric)=}"
        self.metric = metric

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return {
            'score': self.metric(input=y_pred, target=y_true),
        }
