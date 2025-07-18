from typing import Dict, Optional, Any
import torch
from criteria import BaseCriterion
from utils.builders import build_from_config
from utils.input_checks import check_write_file


class MultiTaskCriterion(BaseCriterion):
    __doc__ = r"""This class serves as a container for all criteria needed.
    """

    def __init__(self, criterion_cfgs: Dict[str, Dict[str, Any]], **kwargs) -> None:
        super(MultiTaskCriterion, self).__init__(**kwargs)
        # Build criteria as submodules
        self.task_criteria = torch.nn.ModuleDict({
            task: build_from_config(config=criterion_cfgs[task])
            for task in criterion_cfgs
        })
        assert all(isinstance(criterion, BaseCriterion) for criterion in self.task_criteria.values())
        self.task_names = set(criterion_cfgs.keys())
        self.reset_buffer()

    def reset_buffer(self):
        r"""Reset each criterion.
        """
        if hasattr(self, 'task_criteria'):
            assert isinstance(self.task_criteria, torch.nn.ModuleDict), f"{type(self.task_criteria)=}"
            assert len(self.task_criteria) > 0
            for criterion in self.task_criteria.values():
                criterion.reset_buffer()

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""Call each criterion.
        """
        # input checks
        assert type(y_pred) == type(y_true) == dict, f"{type(y_pred)=}, {type(y_true)=}"
        assert set(y_pred.keys()) & set(y_true.keys()) == set(self.task_names), \
            f"{set(y_pred.keys())=}, {set(y_true.keys())=}, {set(self.task_names)=}"
        # call each task criterion
        losses: Dict[str, torch.Tensor] = dict(
            (task, self.task_criteria[task](y_pred=y_pred[task], y_true=y_true[task]))
            for task in self.task_names
        )
        return losses

    def summarize(self, output_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        r"""Summarize each criterion.
        """
        # call each task buffer summary
        result: Dict[str, torch.Tensor] = {
            task: self.task_criteria[task].summarize(output_path=None)
            for task in self.task_names
        }
        # save to disk
        if output_path is not None:
            check_write_file(path=output_path)
            torch.save(obj=result, f=output_path)
        return result
