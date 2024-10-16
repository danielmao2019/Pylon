from typing import Dict, Optional
import torch
from metrics.base_metric import BaseMetric
from utils.input_checks import check_write_file
from utils.io import save_json
from utils.builder import build_from_config


class MultiTaskMetric(BaseMetric):
    __doc__ = r"""This class serves as a container for all metrics needed.
    """

    def __init__(self, metric_configs: dict) -> None:
        self.task_metrics = {
            task: build_from_config(config=metric_configs[task])
            for task in metric_configs.keys()
        }
        self.task_names = metric_configs.keys()
        super(MultiTaskMetric, self).__init__()

    def reset_buffer(self):
        r"""Reset each metric.
        """
        for metric in self.task_metrics.values():
            metric.reset_buffer()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        r"""Call each metric.
        """
        # input checks
        assert type(y_pred) == type(y_true) == dict, f"{type(y_pred)=}, {type(y_true)=}"
        assert set(y_pred.keys()) & set(y_true.keys()) == set(self.task_names), \
            f"{set(y_pred.keys())=}, {set(y_true.keys())=}, {set(self.task_names)=}"
        # call each task metric
        scores: Dict[str, Dict[str, torch.Tensor]] = {
            task: self.task_metrics[task](y_pred=y_pred[task], y_true=y_true[task])
            for task in self.task_names
        }
        return scores

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, float]:
        r"""Summarize each metric.
        """
        # call each task buffer summary
        result: Dict[str, Dict[str, torch.Tensor]] = {
            task: self.task_metrics[task].summarize(output_path=None)
            for task in self.task_names
        }
        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            save_json(obj=result, filepath=output_path)
        return result
