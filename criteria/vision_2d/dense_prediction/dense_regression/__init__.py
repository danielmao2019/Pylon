"""
CRITERIA.VISION_2D.DENSE_PREDICTION.DENSE_REGRESSION API
"""
from criteria.vision_2d.dense_prediction.dense_regression.base import DenseRegressionCriterion
from criteria.vision_2d.dense_prediction.dense_regression.depth_estimation import DepthEstimationCriterion
from criteria.vision_2d.dense_prediction.dense_regression.normal_estimation import NormalEstimationCriterion
from criteria.vision_2d.dense_prediction.dense_regression.instance_segmentation import InstanceSegmentationCriterion

__all__ = (
    'DenseRegressionCriterion',
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'InstanceSegmentationCriterion',
)
