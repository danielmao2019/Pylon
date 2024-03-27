"""
METRICS API
"""
from metrics.confusion_matrix import ConfusionMatrix
from metrics.depth_estimation import DepthEstimationMetric
from metrics.normal_estimation import NormalEstimationMetric
from metrics.semantic_segmentation import SemanticSegmentationMetric
from metrics.instance_segmentation import InstanceSegmentationMetric


__all__ = (
    'ConfusionMatrix',
    'DepthEstimationMetric',
    'NormalEstimationMetric',
    'SemanticSegmentationMetric',
    'InstanceSegmentationMetric',
)
