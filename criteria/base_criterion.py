from typing import List, Any, Optional
from abc import ABC, abstractmethod
import torch


class BaseCriterion(torch.nn.Module, ABC):

    def __init__(self, use_buffer: bool = True):
        super(BaseCriterion, self).__init__()
        self.use_buffer = use_buffer
        self.reset_buffer()

    def reset_buffer(self):
        if self.use_buffer:
            self.buffer: List[Any] = []

    def add_to_buffer(self, value: torch.Tensor) -> None:
        if self.use_buffer:
            assert hasattr(self, 'buffer')
            assert isinstance(self.buffer, list)
            assert isinstance(value, torch.Tensor), f"{type(value)=}"
            assert value.numel() == 1 and value.ndim == 0, f"{value.shape=}"
            self.buffer.append(value.detach().cpu())
        else:
            assert not hasattr(self, 'buffer')

    @abstractmethod
    def __call__(self, y_pred: Any, y_true: Any) -> Any:
        raise NotImplementedError("Abstract method BaseCriterion.__call__ not implemented.")

    @abstractmethod
    def summarize(self, output_path: Optional[str] = None) -> Any:
        raise NotImplementedError("Abstract method BaseCriterion.summarize not implemented.")
