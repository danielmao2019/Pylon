from typing import List
from abc import abstractmethod
import os
import json
import jsbeautifier
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

    def summarize(self, output_path: str = None) -> torch.Tensor:
        r"""Default summarization: mean of scores across all examples in buffer.
        """
        mean_score = torch.cat(self.buffer).mean()
        assert mean_score.numel() == 1, f"{mean_score.shape=}"
        if output_path is not None and os.path.isfile(output_path):
            with open(output_path, mode='w') as f:
                f.write(jsbeautifier.beautify(json.dumps(mean_score), jsbeautifier.default_options()))
        return mean_score
