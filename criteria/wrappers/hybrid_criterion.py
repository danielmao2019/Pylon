from typing import List, Dict, Any
import torch
from criteria.base_criterion import BaseCriterion
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from utils.builders import build_from_config
from utils.builders.builder import semideepcopy


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

    def __init__(
        self,
        combine: str = 'sum',
        criteria_cfg: List[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        super(HybridCriterion, self).__init__(**kwargs)
        assert combine in self.COMBINE_OPTIONS
        self.combine = combine
        # Build criteria as submodules
        assert criteria_cfg is not None and len(criteria_cfg) > 0
        # Disable buffer for all component criteria by modifying config before building
        modified_configs = []
        for cfg in criteria_cfg:
            if isinstance(cfg, dict) and 'args' in cfg:
                # Make a copy and modify buffer setting
                cfg_copy = semideepcopy(cfg)
                if 'args' not in cfg_copy:
                    cfg_copy['args'] = {}
                cfg_copy['args']['use_buffer'] = False
                modified_configs.append(cfg_copy)
            else:
                # If not a config dict, use as-is (might be pre-built object)
                modified_configs.append(cfg)
        # Build all criteria
        self.criteria = torch.nn.ModuleList([build_from_config(cfg) for cfg in modified_configs])
        # Validate all criteria
        assert all(isinstance(criterion, BaseCriterion) for criterion in self.criteria)
        assert all(not c.use_buffer for c in self.criteria)
        assert all(not hasattr(c, 'buffer') for c in self.criteria)

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for criterion in self.criteria:
            total_loss += criterion(y_pred=y_pred, y_true=y_true)

        if self.combine == 'mean':
            total_loss = total_loss / len(self.criteria)

        return total_loss
