"""
CRITERIA.VISION_2D API
"""
# Base classes
from criteria.vision_2d.dense_prediction.base import DensePredictionCriterion
from criteria.vision_2d.dense_prediction.dense_classification.base import DenseClassificationCriterion
from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion

# Dense Classification
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.spatial_cross_entropy import SpatialCrossEntropyCriterion
from criteria.vision_2d.dense_prediction.dense_classification.iou_loss import IoULoss
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss
from criteria.vision_2d.dense_prediction.dense_classification.ssim_loss import SSIMLoss
from criteria.vision_2d.dense_prediction.dense_classification.ce_dice_loss import CEDiceLoss
from criteria.vision_2d.dense_prediction.dense_classification.focal_dice_loss import FocalDiceLoss

# Dense Regression
from criteria.vision_2d.dense_prediction.dense_regression.depth_estimation import DepthEstimationCriterion
from criteria.vision_2d.dense_prediction.dense_regression.normal_estimation import NormalEstimationCriterion
from criteria.vision_2d.dense_prediction.dense_regression.instance_segmentation import InstanceSegmentationCriterion

# Change Detection Specific
from criteria.vision_2d import change_detection


__all__ = (
    # Base classes
    'DensePredictionCriterion',
    'DenseClassificationCriterion',
    'DenseRegressionCriterion',
    # Dense Classification
    'SemanticSegmentationCriterion',
    'SpatialCrossEntropyCriterion',
    'IoULoss',
    'DiceLoss',
    'SSIMLoss',
    'CEDiceLoss',
    'FocalDiceLoss',
    # Dense Regression
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'InstanceSegmentationCriterion',
    # Change Detection
    'change_detection',
)
