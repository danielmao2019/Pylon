"""
METRICS.WRAPPERS API
"""
from metrics.wrappers.single_task_metric import SingleTaskMetric
from metrics.wrappers.pytorch_metric_wrapper import PyTorchMetricWrapper
from metrics.wrappers.multi_task_metric import MultiTaskMetric


__all__ = (
    'SingleTaskMetric',
    'PyTorchMetricWrapper',
    'MultiTaskMetric',
)
