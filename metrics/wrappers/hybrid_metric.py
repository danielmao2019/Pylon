from typing import List, Dict, Any
import torch
from metrics.base_metric import BaseMetric
from metrics.wrappers.single_task_metric import SingleTaskMetric
from utils.builders import build_from_config


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
        # Disable buffer for all component metrics
        for cfg in metrics_cfg:
            cfg['args']['use_buffer'] = False
        self.metrics = [build_from_config(cfg) for cfg in metrics_cfg]
        assert all(isinstance(metric, BaseMetric) for metric in self.metrics)

    def _compute_score(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        merged_scores = {}
        for metric in self.metrics:
            scores = metric(y_pred=y_pred, y_true=y_true)
            # Assert no key overlaps to avoid ambiguity in merging
            overlapping_keys = set(merged_scores.keys()) & set(scores.keys())
            assert len(overlapping_keys) == 0, f"Key overlap detected: {overlapping_keys}"
            merged_scores.update(scores)
        return merged_scores
