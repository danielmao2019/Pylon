import os
import torch
from runners import BaseTrainer, SupervisedSingleTaskTrainer


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

    # Compare structure
    assert sequential_scores.keys() == parallel_scores.keys(), "Score keys should be identical"
    assert sequential_scores['aggregated'].keys() == parallel_scores['aggregated'].keys(), "Aggregated keys should be identical"
    assert sequential_scores['per_datapoint'].keys() == parallel_scores['per_datapoint'].keys(), "Per-datapoint keys should be identical"

    # Compare aggregated values
    for key in sequential_scores['aggregated']:
        seq_val = sequential_scores['aggregated'][key]
        par_val = parallel_scores['aggregated'][key]
        assert torch.allclose(seq_val, par_val), f"Aggregated {key} differs: {seq_val} vs {par_val}"

    # Compare per-datapoint values
    for key in sequential_scores['per_datapoint']:
        seq_vals = sequential_scores['per_datapoint'][key]
        par_vals = parallel_scores['per_datapoint'][key]
        assert torch.allclose(seq_vals, par_vals), f"Per-datapoint {key} differs: {seq_vals} vs {par_vals}"
