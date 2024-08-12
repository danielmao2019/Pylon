"""
CRITERIA.VISION_2D API
"""
from criteria.vision_2d.depth_estimation_criterion import DepthEstimationCriterion
from criteria.vision_2d.normal_estimation_criterion import NormalEstimationCriterion
from criteria.vision_2d.semantic_segmentation_criterion import SemanticSegmentationCriterion
from criteria.vision_2d.instance_segmentation_criterion import InstanceSegmentationCriterion


__all__ = (
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'SemanticSegmentationCriterion',
    'InstanceSegmentationCriterion',
)
