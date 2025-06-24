from typing import List, Dict, Any
import torch
from metrics.base_metric import BaseMetric
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.builders import build_from_config
from utils.builders.builder import semideepcopy


class HybridMetric(SingleTaskMetric):
    """
    A wrapper class that combines multiple metrics based on configuration dictionaries.

    This class allows for flexible combination of multiple metrics by providing
    their configurations. Each metric is built using build_from_config and then
    their scores are merged into one dictionary, asserting no key overlaps.

    Args:
        metrics_cfg (List[Dict]): List of metric configuration dictionaries
    """

    def __init__(
        self,
        metrics_cfg: List[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        super(HybridMetric, self).__init__(**kwargs)
        # Build metrics
        assert metrics_cfg is not None and len(metrics_cfg) > 0
        # Disable buffer for all component metrics by modifying config before building
        modified_configs = []
        for cfg in metrics_cfg:
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
        # Build all metrics
        self.metrics = [build_from_config(cfg) for cfg in modified_configs]
        # Validate all metrics
        assert all(isinstance(m, BaseMetric) for m in self.metrics)
        assert all(not m.use_buffer for m in self.metrics)
        assert all(not hasattr(m, 'buffer') for m in self.metrics)

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        merged_scores = {}
        for metric in self.metrics:
            scores = metric(y_pred=y_pred, y_true=y_true)
            # Assert no key overlaps to avoid ambiguity in merging
            overlapping_keys = set(merged_scores.keys()) & set(scores.keys())
            assert len(overlapping_keys) == 0, f"Key overlap detected: {overlapping_keys}"
            merged_scores.update(scores)
        return merged_scores
