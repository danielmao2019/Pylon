from typing import List, Dict, Union
import torch
from metrics.base_metric import BaseMetric
from utils.input_checks.str_types import check_write_file
from utils.io.json import save_json
from utils.ops.dict_as_tensor import transpose_buffer


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
        scores: Dict[str, torch.Tensor] = self._compute_score(y_pred=y_pred, y_true=y_true)
        assert isinstance(scores, dict), f"{type(scores)=}"
        assert all([isinstance(k, str) for k in scores.keys()]), \
            f"{{{', '.join([f'{k}: {type(k)}' for k in scores.keys()])}}}"
        assert all([isinstance(v, torch.Tensor) for v in scores.values()]), \
            f"{{{', '.join([f'{k}: {type(v)}' for k, v in scores.items()])}}}"
        self.add_to_buffer(scores)
        return scores

    def summarize(self, output_path: str = None) -> Dict[str, torch.Tensor]:
        r"""This method averages scores across all data points in buffer.
        """
        self._buffer_queue.join()  # Wait for all items to be processed
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
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
            assert key_scores.ndim == 1, f"{key=}, {key_scores.shape=}"
            # Store per-datapoint values
            result["per_datapoint"][key] = key_scores
            # Store aggregated value
            result["aggregated"][key] = key_scores.mean()

        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
