"""
CRITERIA.VISION_2D.DENSE_PREDICTION.DENSE_CLASSIFICATION API
"""
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.spatial_cross_entropy import SpatialCrossEntropyCriterion
from criteria.vision_2d.dense_prediction.dense_classification.iou_loss import IoULoss
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss
from criteria.vision_2d.dense_prediction.dense_classification.ssim_loss import SSIMLoss
from criteria.vision_2d.dense_prediction.dense_classification.ce_dice_loss import CEDiceLoss

__all__ = (
    'DenseClassificationCriterion',
    'SemanticSegmentationCriterion',
    'SpatialCrossEntropyCriterion',
    'IoULoss',
    'DiceLoss',
    'SSIMLoss',
    'CEDiceLoss',
)
