from typing import List
from abc import abstractmethod
import torch


class BaseMetric:

    def __init__(self):
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[torch.Tensor] = []

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Returns:
            score (torch.Tensor): a scalar tensor for the score on the current batch.
        """
        raise NotImplementedError("__call__ method not implemented for base class.")

    def summarize(self) -> torch.Tensor:
        r"""Default summarization: mean of scores across all examples in buffer.
        """
        summary = torch.cat(self.buffer).mean()
        assert summary.numel() == 1, f"{summary.shape=}"
        return summary
