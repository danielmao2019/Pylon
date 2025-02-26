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
from criteria.vision_2d.semantic_map.ce_dice_loss import CEDiceLoss
# Change Detection Specific
from criteria.vision_2d import change_detection


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
    'CEDiceLoss',
    # Change Detection Specific
    'change_detection',
)
