from typing import List, Dict, Union
from abc import ABC, abstractmethod
import os
import json
import jsbeautifier
import torch


class BaseMetric(ABC):

    def __init__(self):
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[torch.Tensor] = []

    @abstractmethod
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Returns:
            score (torch.Tensor or Dict[str, torch.Tensor]): a scalar tensor for single score
                or dictionary of scalar tensors for multiple scores.
        """
        raise NotImplementedError("[ERROR] __call__ not implemented for abstract base class.")

    @staticmethod
    def reduce(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Reduction is needed when comparing checkpoints.
        Default reduction: mean of scores across all metrics.
        """
        for val in scores.values():
            assert type(val) == torch.Tensor
            assert val.numel() == 1
        reduced = torch.cat(list(scores.values()))
        assert reduced.shape == (len(scores),), f"{reduced.shape=}"
        return reduced.mean()

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""Default summary: mean of scores across all examples in buffer.
        """
        summary: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            if type(self.buffer[0]) == torch.Tensor:
                score = torch.cat(self.buffer)
                assert len(score.shape) == 1
                summary['score'] = score.mean()
                summary['reduced'] = summary['score'].clone()
            elif type(self.buffer[0]) == dict:
                keys = self.buffer[0].keys()
                for scores in self.buffer:
                    assert scores.keys() == keys
                    for key in keys:
                        summary[f"score_{key}"] = summary.get(f"score_{key}", []) + [scores[key]]
                for key in keys:
                    score_key = torch.cat(summary[f"score_{key}"])
                    assert len(score_key.shape) == 1
                    summary[f"score_{key}"] = score_key.mean()
                summary['reduced'] = self.reduce(summary)
            else:
                raise TypeError(f"[ERROR] Unrecognized type {type(self.buffer[0])}.")
        if output_path is not None and os.path.isfile(output_path):
            with open(output_path, mode='w') as f:
                f.write(jsbeautifier.beautify(json.dumps(summary), jsbeautifier.default_options()))
        return summary
