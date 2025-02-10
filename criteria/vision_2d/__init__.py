"""
CRITERIA.VISION_2D API
"""
from criteria.vision_2d.depth_estimation_criterion import DepthEstimationCriterion
from criteria.vision_2d.normal_estimation_criterion import NormalEstimationCriterion
from criteria.vision_2d.spatial_cross_entropy_criterion import SpatialCrossEntropyCriterion
from criteria.vision_2d.semantic_segmentation_criterion import SemanticSegmentationCriterion
from criteria.vision_2d.instance_segmentation_criterion import InstanceSegmentationCriterion
from criteria.vision_2d.dice_loss import DiceLoss
# Change Detection Specific
from criteria.vision_2d.change_detection.symmetric_change_detection_criterion import SymmetricChangeDetectionCriterion
from criteria.vision_2d.change_detection.change_star_criterion import ChangeStarCriterion
from criteria.vision_2d.change_detection.ppsl_criterion import PPSLCriterion


__all__ = (
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'SpatialCrossEntropyCriterion',
    'SemanticSegmentationCriterion',
    'InstanceSegmentationCriterion',
    'DiceLoss',
    # Change Detection Specific
    'SymmetricChangeDetectionCriterion',
    'ChangeStarCriterion',
    'PPSLCriterion',
)
