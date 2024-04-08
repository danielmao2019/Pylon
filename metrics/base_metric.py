from typing import List, Dict, Union, Any
from abc import ABC, abstractmethod
import os
import json
import jsbeautifier
import torch
from utils.ops import transpose_buffer


class BaseMetric(ABC):

    def __init__(self):
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[Any] = []

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
        result: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            if type(self.buffer[0]) == torch.Tensor:
                score = torch.stack(self.buffer)
                assert len(score.shape) == 1
                result['score'] = score.mean()
                result['reduced'] = result['score'].clone()
            elif type(self.buffer[0]) == dict:
                buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
                for key in buffer:
                    key_scores = torch.stack(buffer[key])
                    assert len(key_scores.shape) == 1
                    result[f"score_{key}"] = key_scores.mean()
                result['reduced'] = self.reduce(result)
            else:
                raise TypeError(f"[ERROR] Unrecognized type {type(self.buffer[0])}.")
        if output_path is not None and os.path.isfile(output_path):
            with open(output_path, mode='w') as f:
                f.write(jsbeautifier.beautify(json.dumps(result), jsbeautifier.default_options()))
        return result
