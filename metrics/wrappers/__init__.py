"""
METRICS.WRAPPERS API
"""

from metrics.wrappers.hybrid_metric import HybridMetric
from metrics.wrappers.multi_task_metric import MultiTaskMetric
from metrics.wrappers.pytorch_metric_wrapper import PyTorchMetricWrapper
from metrics.wrappers.single_task_metric import SingleTaskMetric

__all__ = (
    'SingleTaskMetric',
    'PyTorchMetricWrapper',
    'MultiTaskMetric',
    'HybridMetric',
)
