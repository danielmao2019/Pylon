"""
RUNNERS API
"""
from runners.base_trainer import BaseTrainer
from runners.supervised_single_task_trainer import SupervisedSingleTaskTrainer
from runners.supervised_multi_task_trainer import SupervisedMultiTaskTrainer
from runners import gan_trainers


__all__ = (
    'BaseTrainer',
    'SupervisedSingleTaskTrainer',
    'SupervisedMultiTaskTrainer',
    'gan_trainers',
)
