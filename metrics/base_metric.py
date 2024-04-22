from typing import List, Dict, Union, Any
from abc import ABC, abstractmethod
import os
import torch
from utils.ops import transpose_buffer
from utils.io import save_json


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
            assert len(val.shape) == 0 and val.numel() == 1, f"{val.shape=}, {val.numel()=}"
        reduced = torch.stack(list(scores.values()))
        assert reduced.shape == (len(scores),), f"{reduced.shape=}"
        return reduced.mean()

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""Default summary: mean of scores across all examples in buffer.
        """
        if output_path is not None:
            assert type(output_path) == str, f"{type(output_path)=}"
            assert os.path.isdir(os.path.dirname(output_path)), f"{output_path=}"
        result: Dict[str, torch.Tensor] = {}
        if len(self.buffer) == 0:
            return result
        if type(self.buffer[0]) == torch.Tensor:
            scores = torch.stack(self.buffer)
            assert scores.shape == (len(self.buffer),), f"{scores.shape=}"
            result['score'] = scores.mean()
        elif type(self.buffer[0]) == dict:
            buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
            for key in buffer:
                key_scores = torch.stack(buffer[key])
                assert key_scores.shape == (len(buffer[key]),), f"{key_scores.shape=}"
                result[f"score_{key}"] = key_scores.mean()
        else:
            raise TypeError(f"[ERROR] Unrecognized type {type(self.buffer[0])}.")
        assert 'reduced' not in result, f"{result.keys()=}"
        result['reduced'] = self.reduce(result)
        if output_path is not None:
            save_json(obj=result, filepath=output_path)
        return result
