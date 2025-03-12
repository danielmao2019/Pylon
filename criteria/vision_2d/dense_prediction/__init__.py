"""
CRITERIA.VISION_2D.DENSE_PREDICTION API
"""
from criteria.vision_2d.dense_prediction.base import DensePredictionCriterion
from criteria.vision_2d.dense_prediction import dense_classification
from criteria.vision_2d.dense_prediction import dense_regression

__all__ = (
    'DensePredictionCriterion',
    'dense_classification',
    'dense_regression',
)
