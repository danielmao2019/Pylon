"""
RUNNERS API
"""
from runners.base_trainer import BaseTrainer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from runners.supervised_multi_task_trainer import SupervisedMultiTaskTrainer
from runners import gan_trainers
from runners.multi_val_dataset_trainer import MultiValDatasetTrainer
from runners.base_evaluator import BaseEvaluator
from runners.multi_stage_trainer import MultiStageTrainer


__all__ = (
    'BaseTrainer',
    'SupervisedSingleTaskTrainer',
    'SupervisedMultiTaskTrainer',
    'gan_trainers',
    'MultiValDatasetTrainer',
    'BaseEvaluator',
    'MultiStageTrainer',
)
