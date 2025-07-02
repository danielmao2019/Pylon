import os
import time
import torch
from runners import BaseEvaluator


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
            self._eval_step(dp, flush_prefix=f"Evaluation [Iteration {idx}/{len(self.eval_dataloader)}].")

        # after validation loop
        self._after_eval_loop_()
        # log time
        self.logger.info(f"Evaluation epoch time: {round(time.time() - start_time, 2)} seconds.")


def test_sequential_vs_parallel_evaluation(test_dir, evaluator_cfg):
    """Test that sequential and parallel evaluation produce identical results."""
    # Create directories for sequential and parallel evaluation
    sequential_dir = os.path.join(test_dir, "sequential_eval")
    parallel_dir = os.path.join(test_dir, "parallel_eval")
    os.makedirs(sequential_dir, exist_ok=True)
    os.makedirs(parallel_dir, exist_ok=True)

    # Create configurations
    sequential_config = {
        **evaluator_cfg,
        'work_dir': sequential_dir,
    }

    parallel_config = {
        **evaluator_cfg,
        'work_dir': parallel_dir,
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
