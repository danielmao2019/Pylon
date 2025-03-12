"""
CRITERIA.WRAPPERS API
"""
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.spatial_pytorch_criterion_wrapper import SpatialPyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion
from criteria.wrappers.multi_task_criterion import MultiTaskCriterion
from criteria.wrappers.dense_prediction_criterion import DensePredictionCriterion
from criteria.wrappers.dense_classification_criterion import DenseClassificationCriterion
from criteria.wrappers.dense_regression_criterion import DenseRegressionCriterion


__all__ = (
    'SingleTaskCriterion',
    'PyTorchCriterionWrapper',
    'SpatialPyTorchCriterionWrapper',
    'AuxiliaryOutputsCriterion',
    'MultiTaskCriterion',
    'DensePredictionCriterion',
    'DenseClassificationCriterion',
    'DenseRegressionCriterion',
)
