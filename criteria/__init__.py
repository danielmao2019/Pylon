"""
CRITERIA API
"""
from criteria.base_criterion import BaseCriterion
from criteria.pytorch_criteria_wrapper import PyTorchCriterionWrapper
from criteria.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion
from criteria.multi_task_criterion import MultiTaskCriterion
from criteria.depth_estimation_criterion import DepthEstimationCriterion
from criteria.normal_estimation_criterion import NormalEstimationCriterion
from criteria.semantic_segmentation_criterion import SemanticSegmentationCriterion
from criteria.instance_segmentation_criterion import InstanceSegmentationCriterion
from criteria.ccdm_criterion import CCDMCriterion


__all__ = (
    'BaseCriterion',
    'PyTorchCriterionWrapper',
    'AuxiliaryOutputsCriterion',
    'MultiTaskCriterion',
    'DepthEstimationCriterion',
    'NormalEstimationCriterion',
    'SemanticSegmentationCriterion',
    'InstanceSegmentationCriterion',
    'CCDMCriterion',
)
