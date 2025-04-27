from typing import List, Any, Optional
from abc import ABC, abstractmethod
import torch


class BaseCriterion(torch.nn.Module, ABC):

    def __init__(self, use_buffer: bool = True):
        super(BaseCriterion, self).__init__()
        self.use_buffer = use_buffer
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[Any] = [] if self.use_buffer else None

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("Abstract method BaseCriterion.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseCriterion.summarize not implemented.")
