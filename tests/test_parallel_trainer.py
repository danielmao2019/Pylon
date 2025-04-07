import os
import sys
from runners import BaseTrainer, SupervisedSingleTaskTrainer

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SimpleTrainer(BaseTrainer):
    """A simplified trainer for testing purposes."""
    
    def _init_optimizer_(self) -> None:
        pass
    
    def _init_scheduler_(self) -> None:
        pass
    
    def _set_gradients_(self, dp: dict) -> None:
        pass


def test_sequential_vs_parallel_validation(test_dir, trainer_cfg):
    """Test that sequential and parallel validation produce identical results."""
    # Create directories for sequential and parallel validation
    sequential_dir = os.path.join(test_dir, "sequential_val")
    parallel_dir = os.path.join(test_dir, "parallel_val")
    os.makedirs(sequential_dir, exist_ok=True)
    os.makedirs(parallel_dir, exist_ok=True)

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
