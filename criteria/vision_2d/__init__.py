"""
CRITERIA.VISION_2D API
"""
from criteria.vision_2d.depth_estimation_criterion import DepthEstimationCriterion
from criteria.vision_2d.normal_estimation_criterion import NormalEstimationCriterion
from criteria.vision_2d.instance_segmentation_criterion import InstanceSegmentationCriterion
# Semantic Map
from criteria.vision_2d.semantic_map.semantic_map_base_criterion import SemanticMapBaseCriterion
from criteria.vision_2d.semantic_map.spatial_cross_entropy_criterion import SpatialCrossEntropyCriterion
from criteria.vision_2d.semantic_map.semantic_segmentation_criterion import SemanticSegmentationCriterion
from criteria.vision_2d.semantic_map.iou_loss import IoULoss
from criteria.vision_2d.semantic_map.dice_loss import DiceLoss
from criteria.vision_2d.semantic_map.ssim_loss import SSIMLoss
# Change Detection Specific
from criteria.vision_2d.change_detection.symmetric_change_detection_criterion import SymmetricChangeDetectionCriterion
from criteria.vision_2d.change_detection.change_former_criterion import ChangeFormerCriterion
from criteria.vision_2d.change_detection.ftn_criterion import FTNCriterion
from criteria.vision_2d.change_detection.change_star_criterion import ChangeStarCriterion
from criteria.vision_2d.change_detection.ppsl_criterion import PPSLCriterion
from criteria.vision_2d.dice_loss import DiceLoss
from criteria.vision_2d.change_detection.snunet_criterion import SNUNetCDCriterion


__all__ = (
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'InstanceSegmentationCriterion',
    # Semantic Map
    'SemanticMapBaseCriterion',
    'SpatialCrossEntropyCriterion',
    'SemanticSegmentationCriterion',
    'IoULoss',
    'DiceLoss',
    'SSIMLoss',
    # Change Detection Specific
    'SymmetricChangeDetectionCriterion',
    'ChangeFormerCriterion',
    'FTNCriterion',
    'ChangeStarCriterion',
    'PPSLCriterion',
    'DiceLoss',
    'SNUNetCDCriterion',
)
