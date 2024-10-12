from typing import List, Dict, Union, Any
import torch
from utils.input_checks import check_write_file
from utils.ops import transpose_buffer
from utils.io import save_json


class BaseMetric:

    def __init__(self):
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer: List[Any] = []

    def __call__(
        self,
        y_pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Default __call__ method. This method assumes `_compute_score_` is implemented and both y_pred
        and y_true are either tensors or dictionaries of exactly one key-val pair.
        """
        assert hasattr(self, '_compute_score_') and callable(self._compute_score_)
        # input checks
        if type(y_pred) == dict:
            assert len(y_pred) == 1, f"{y_pred.keys()=}"
            y_pred = list(y_pred.values())[0]
        assert type(y_pred) == torch.Tensor, f"{type(y_pred)=}"
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        assert type(y_true) == torch.Tensor, f"{type(y_true)=}"
        # compute score
        score = self._compute_score_(y_pred=y_pred, y_true=y_true)
        score = score.detach().cpu()
        self.buffer.append(score)
        return score

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
            check_write_file(path=output_path)
        result: Dict[str, torch.Tensor] = {}
        if len(self.buffer) != 0:
            if type(self.buffer[0]) == torch.Tensor:
                scores = torch.stack(self.buffer)
                assert scores.shape == (len(self.buffer),), f"{scores.shape=}"
                result['score'] = scores.mean()
            elif type(self.buffer[0]) == dict:
                buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
                for key in buffer:
                    key_scores = torch.stack(buffer[key], dim=0)
                    assert key_scores.shape == (len(buffer[key]),), f"{key_scores.shape=}"
                    result[f"score_{key}"] = key_scores.mean()
            else:
                raise TypeError(f"[ERROR] Unrecognized type {type(self.buffer[0])}.")
            # log reduction
            assert 'reduced' not in result, f"{result.keys()=}"
            result['reduced'] = self.reduce(result)
        if output_path is not None:
            save_json(obj=result, filepath=output_path)
        return result
