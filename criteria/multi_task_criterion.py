from typing import Dict, Optional
import torch
from .base_criterion import BaseCriterion
from utils.input_checks import check_write_file
from utils.builder import build_from_config


class MultiTaskCriterion(BaseCriterion):
    __doc__ = r"""This class serves as a container for all criteria needed.
    """

    def __init__(self, criterion_configs: dict) -> None:
        self.task_criteria = {
            task: build_from_config(config=criterion_configs[task])
            for task in criterion_configs.keys()
        }
        self.task_names = criterion_configs.keys()
        super(MultiTaskCriterion, self).__init__()

    def reset_buffer(self):
        r"""Reset each criterion.
        """
        for criterion in self.task_criteria.values():
            criterion.reset_buffer()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""Call each criterion.
        """
        # input checks
        assert type(y_pred) == type(y_true) == dict, f"{type(y_pred)=}, {type(y_true)=}"
        assert set(y_pred.keys()) & set(y_true.keys()) == set(self.task_names), \
            f"{set(y_pred.keys())=}, {set(y_true.keys())=}, {set(self.task_names)=}"
        # compute loss for each task
        losses: Dict[str, torch.Tensor] = dict(
            (task, self.task_criteria[task](y_pred=y_pred[task], y_true=y_true[task]))
            for task in self.task_names
        )
        return losses

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        r"""Summarize each criterion.
        """
        if output_path is not None:
            check_write_file(path=output_path)
        # call summarize method of each criterion
        result: Dict[str, torch.Tensor] = {
            task: self.task_criteria[task].summarize(output_path=None)
            for task in self.task_names
        }
        # save to disk
        if output_path is not None:
            torch.save(obj=result, f=output_path)
        return result
