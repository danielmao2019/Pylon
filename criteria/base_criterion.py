from typing import List
from abc import ABC, abstractmethod
import os
import torch


class BaseCriterion(ABC):

    def __init__(self):
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[torch.Tensor] = []

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Returns:
            loss (torch.Tensor): a scalar tensor for the loss on the current batch.
        """
        raise NotImplementedError("[ERROR] __call__ not implemented for abstract base class.")

    def summarize(self, output_path: str = None) -> torch.Tensor:
        r"""Default summary: trajectory of losses across all examples in buffer.
        """
        summary = torch.cat(self.buffer)
        assert len(summary.shape) == 1, f"{summary.shape=}"
        if output_path is not None and os.path.isfile(output_path):
            torch.save(obj=summary, f=output_path)
        return summary
