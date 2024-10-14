from typing import List, Any, Optional
from abc import ABC, abstractmethod


class BaseMetric(ABC):

    def __init__(self):
        super(BaseMetric, self).__init__()
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[Any] = []

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("BaseMetric.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("BaseMetric.summarize not implemented.")
