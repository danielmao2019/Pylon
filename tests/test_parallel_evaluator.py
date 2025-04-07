import os
import sys
import pytest
import torch
import json
import time
from typing import Dict, Any
from utils.logging import Logger
from runners import BaseEvaluator
from metrics.base_metric import BaseMetric
from utils.builders import build_from_config

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SequentialEvaluator(BaseEvaluator):
    """An evaluator that runs evaluation sequentially for testing purposes."""
    
    def _eval_epoch_(self) -> None:
        """Run evaluation sequentially."""
        assert self.eval_dataloader and self.model
        # init time
        start_time = time.time()
        # do validation loop
        self.model.eval()
        self.metric.reset_buffer()
        
        # Process evaluation data sequentially
        for idx, dp in enumerate(self.eval_dataloader):
            self._process_eval_batch(dp)
            self.logger.flush(prefix=f"Evaluation [Iteration {idx}/{len(self.eval_dataloader)}].")
        
        # after validation loop
        self._after_eval_loop_()
        # log time
        self.logger.info(f"Evaluation epoch time: {round(time.time() - start_time, 2)} seconds.")

    def _init_eval_dataloader(self) -> None:
        """Initialize evaluation dataloader."""
        if 'eval_dataloader' in self.config:
            self.eval_dataloader = build_from_config(self.config['eval_dataloader'])
        else:
            self.eval_dataloader = None


def test_sequential_vs_parallel_evaluation(test_dir, evaluator_cfg):
    """Test that sequential and parallel evaluation produce identical results."""
    # Create directories for sequential and parallel evaluation
    sequential_dir = os.path.join(test_dir, "sequential_eval")
    parallel_dir = os.path.join(test_dir, "parallel_eval")
    os.makedirs(sequential_dir, exist_ok=True)
    os.makedirs(parallel_dir, exist_ok=True)

    # Create log files
    sequential_log = os.path.join(sequential_dir, "eval_0.log")
    parallel_log = os.path.join(parallel_dir, "eval_0.log")

    # Create configurations
    sequential_config = {
        **evaluator_cfg,
        'work_dir': sequential_dir,
        'eval_dataset': evaluator_cfg['eval_dataloader']['args']['dataset']
    }

    parallel_config = {
        **evaluator_cfg,
        'work_dir': parallel_dir,
        'eval_dataset': evaluator_cfg['eval_dataloader']['args']['dataset']
    }

    # Run sequential evaluation
    sequential_evaluator = SequentialEvaluator(sequential_config)
    sequential_evaluator._init_components_()
    sequential_evaluator._eval_epoch_()

    # Run parallel evaluation
    parallel_evaluator = BaseEvaluator(parallel_config)
    parallel_evaluator._init_components_()
    parallel_evaluator._eval_epoch_()

    # Compare results
    sequential_scores = sequential_evaluator.metric.summarize()
    parallel_scores = parallel_evaluator.metric.summarize()

    assert sequential_scores == parallel_scores, "Sequential and parallel evaluation results should be identical"
