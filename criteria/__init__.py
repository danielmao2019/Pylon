"""
CRITERIA API
"""
from criteria.depth_estimation import DepthEstimationLoss
from criteria.normal_estimation import NormalEstimationLoss
from criteria.semantic_segmentation import SemanticSegmentationLoss
from criteria.instance_segmentation import InstanceSegmentationLoss


__all__ = (
    'DepthEstimationLoss',
    'NormalEstimationLoss',
    'SemanticSegmentationLoss',
    'InstanceSegmentationLoss',
)
