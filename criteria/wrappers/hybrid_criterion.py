from typing import List, Dict, Any
import torch
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from utils.builders import build_from_config


class HybridCriterion(SingleTaskCriterion):
    """
    A wrapper class that combines multiple criteria based on configuration dictionaries.

    This class allows for flexible combination of multiple loss functions by providing
    their configurations. Each criterion is built using build_from_config and then
    combined according to the specified method.

    Args:
        combine (str): How to combine the losses ('sum' or 'mean')
        criteria_cfg (List[Dict]): List of criterion configuration dictionaries
    """

    COMBINE_OPTIONS = {'mean', 'sum'}

    def __init__(self, combine: str = 'sum', criteria_cfg: List[Dict[str, Dict[str, Any]]] = None) -> None:
        super(HybridCriterion, self).__init__()
        assert combine in self.COMBINE_OPTIONS
        self.combine = combine
        assert criteria_cfg is not None and len(criteria_cfg) > 0
        # Register criteria as submodules using ModuleList
        self.criteria = torch.nn.ModuleList([build_from_config(cfg) for cfg in criteria_cfg])

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for criterion in self.criteria:
            total_loss += criterion(y_pred=y_pred, y_true=y_true)

        if self.combine == 'mean':
            total_loss = total_loss / len(self.criteria)

        return total_loss
