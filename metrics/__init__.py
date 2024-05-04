"""
METRICS API
"""
from metrics.base_metric import BaseMetric
from metrics.multi_task_metric import MultiTaskMetric
from metrics.confusion_matrix import ConfusionMatrix
from metrics.depth_estimation_metric import DepthEstimationMetric
from metrics.normal_estimation_metric import NormalEstimationMetric
from metrics.object_detection_metric import ObjectDetectionMetric
from metrics.semantic_segmentation_metric import SemanticSegmentationMetric
from metrics.instance_segmentation_metric import InstanceSegmentationMetric


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
