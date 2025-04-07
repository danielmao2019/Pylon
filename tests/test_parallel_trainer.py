import os
import sys
import pytest
import torch
import json
import time
from typing import Dict, Any
from utils.logging import Logger
from runners import BaseTrainer, SupervisedSingleTaskTrainer
from metrics.base_metric import BaseMetric
from utils.builders import build_from_config

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SimpleTrainer(BaseTrainer):
    """A simplified trainer for testing purposes."""
    
    def _init_optimizer_(self) -> None:
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def _init_scheduler_(self) -> None:
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
    
    def _set_gradients_(self, dp: dict) -> None:
        dp['losses'].backward()


def test_sequential_vs_parallel_validation(test_dir, trainer_cfg):
    """Test that sequential and parallel validation produce identical results."""
    # Create directories for sequential and parallel validation
    sequential_dir = os.path.join(test_dir, "sequential_val")
    parallel_dir = os.path.join(test_dir, "parallel_val")
    os.makedirs(sequential_dir, exist_ok=True)
    os.makedirs(parallel_dir, exist_ok=True)

    # Create log files
    sequential_log = os.path.join(sequential_dir, "train_val_0.log")
    parallel_log = os.path.join(parallel_dir, "train_val_0.log")

    # Create configurations
    sequential_config = {
        **trainer_cfg,
        'work_dir': sequential_dir
    }

    parallel_config = {
        **trainer_cfg,
        'work_dir': parallel_dir
    }

    # Run sequential validation
    sequential_trainer = SimpleTrainer(sequential_config)
    sequential_trainer._init_components_()
    sequential_trainer._val_epoch_()

    # Run parallel validation
    parallel_trainer = SupervisedSingleTaskTrainer(parallel_config)
    parallel_trainer._init_components_()
    parallel_trainer._val_epoch_()

    # Compare results
    sequential_scores = sequential_trainer.metric.summarize()
    parallel_scores = parallel_trainer.metric.summarize()

    assert sequential_scores == parallel_scores, "Sequential and parallel validation results should be identical"
