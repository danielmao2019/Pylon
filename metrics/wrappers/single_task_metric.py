from typing import List, Dict, Union
import torch
from metrics import BaseMetric
from utils.input_checks import check_write_file
from utils.ops import apply_tensor_op, transpose_buffer
from utils.io import save_json


class SingleTaskMetric(BaseMetric):

    DIRECTION: int

    def __call__(
        self,
        y_pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        r"""This method assumes `_compute_score` is implemented and both y_pred
        and y_true are either tensors or dictionaries of exactly one key-val pair.
        """
        assert hasattr(self, '_compute_score') and callable(self._compute_score)
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
        score: Dict[str, torch.Tensor] = self._compute_score(y_pred=y_pred, y_true=y_true)
        assert type(score) == dict, f"{type(score)=}"
        assert all([isinstance(k, str) for k in score.keys()])
        assert all([isinstance(v, torch.Tensor) for v in score.values()])
        score = apply_tensor_op(func=lambda x: x.detach().cpu(), inputs=score)
        # log score
        self.buffer.append(score)
        return score

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""This method averages scores across all data points in buffer.
        """
        assert len(self.buffer) != 0
        buffer: Dict[str, List[torch.Tensor]] = transpose_buffer(self.buffer)
        # summarize scores
        result: Dict[str, Dict[str, torch.Tensor]] = {
            "aggregated": {},
            "per_datapoint": {},
        }

        # For each metric, store both the per-datapoint values and compute the mean
        for key in buffer:
            key_scores = torch.stack(buffer[key], dim=0)
            assert key_scores.ndim == 1, f"{key_scores.shape=}"
            # Store per-datapoint values
            result["per_datapoint"][key] = key_scores
            # Store aggregated value
            result["aggregated"][key] = key_scores.mean()

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
