"""
CRITERIA.WRAPPERS API
"""
from criteria.wrappers.pytorch_criteria_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion
from criteria.wrappers.multi_task_criterion import MultiTaskCriterion


__all__ = (
    'PyTorchCriterionWrapper',
    'AuxiliaryOutputsCriterion',
    'MultiTaskCriterion',
)
