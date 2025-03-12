"""
CRITERIA.VISION_2D API
"""
from criteria.vision_2d.dense_prediction import (
    DensePredictionCriterion,
    dense_classification,
    dense_regression,
)
from criteria.vision_2d.dense_prediction.dense_classification import (
    DenseClassificationCriterion,
    SemanticSegmentationCriterion,
    SpatialCrossEntropyCriterion,
    IoULoss,
    DiceLoss,
    SSIMLoss,
    CEDiceLoss,
)
from criteria.vision_2d.dense_prediction.dense_regression import (
    DenseRegressionCriterion,
    DepthEstimationCriterion,
    NormalEstimationCriterion,
    InstanceSegmentationCriterion,
)
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
    # Dense Regression
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'InstanceSegmentationCriterion',
    # Submodules
    'dense_classification',
    'dense_regression',
    'change_detection',
)
