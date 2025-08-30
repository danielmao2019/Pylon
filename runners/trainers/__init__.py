"""
RUNNERS.TRAINERS API
"""

from runners.trainers.base_trainer import BaseTrainer
from runners.trainers.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from runners.trainers.supervised_multi_task_trainer import SupervisedMultiTaskTrainer
from runners.trainers.multi_val_dataset_trainer import MultiValDatasetTrainer
from runners.trainers.multi_stage_trainer import MultiStageTrainer
from runners.trainers import gan_trainers
from runners.trainers import pcr_trainers


__all__ = [
    'BaseTrainer',
    'SupervisedSingleTaskTrainer',
    'SupervisedMultiTaskTrainer',
    'MultiValDatasetTrainer',
    'MultiStageTrainer',
    'gan_trainers',
    'pcr_trainers',
]
