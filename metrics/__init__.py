"""
METRICS API
"""
from metrics.base_metric import BaseMetric
from metrics.multi_task_metric import MultiTaskMetric
from metrics.confusion_matrix import ConfusionMatrix
from metrics.depth_estimation import DepthEstimationMetric
from metrics.normal_estimation import NormalEstimationMetric
from metrics.object_detection import ObjectDetectionMetric
from metrics.semantic_segmentation import SemanticSegmentationMetric
from metrics.instance_segmentation import InstanceSegmentationMetric


__all__ = (
    'BaseMetric',
    'MultiTaskMetric',
    'ConfusionMatrix',
    'DepthEstimationMetric',
    'NormalEstimationMetric',
    'ObjectDetectionMetric',
    'SemanticSegmentationMetric',
    'InstanceSegmentationMetric',
)
