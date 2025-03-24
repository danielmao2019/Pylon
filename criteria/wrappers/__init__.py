"""
CRITERIA.WRAPPERS API
"""
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.spatial_pytorch_criterion_wrapper import SpatialPyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion
from criteria.wrappers.multi_task_criterion import MultiTaskCriterion


__all__ = (
    'SingleTaskCriterion',
    'PyTorchCriterionWrapper',
    'SpatialPyTorchCriterionWrapper',
    'HybridCriterion',
    'AuxiliaryOutputsCriterion',
    'MultiTaskCriterion',
)
