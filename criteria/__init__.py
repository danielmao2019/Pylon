"""
CRITERIA API
"""
from criteria.base_criterion import BaseCriterion
from criteria.depth_estimation_criterion import DepthEstimationCriterion
from criteria.normal_estimation_criterion import NormalEstimationCriterion
from criteria.semantic_segmentation_criterion import SemanticSegmentationCriterion
from criteria.instance_segmentation_criterion import InstanceSegmentationCriterion


__all__ = (
    'BaseCriterion',
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'SemanticSegmentationCriterion',
    'InstanceSegmentationCriterion',
)
