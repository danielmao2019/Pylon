"""
Test core early stopping functionality.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs
- Edge case testing
- Invalid input testing
- Determinism testing
"""
from typing import Dict
import os
import json
import tempfile
import pytest
import torch
from runners.early_stopping import EarlyStopping
from metrics.wrappers.single_task_metric import SingleTaskMetric


class TestMetric(SingleTaskMetric):
    """Test metric with controllable DIRECTIONS."""
    DIRECTIONS = {"loss": -1}  # Lower is better (loss metric)

    def _compute_score(self, y_pred, y_true):
        return {"loss": torch.tensor(0.5)}


def create_epoch_with_scores(work_dir: str, epoch_idx: int, scores: Dict[str, float]):
    """Helper to create a completed epoch with validation scores."""
    epoch_dir = os.path.join(work_dir, f"epoch_{epoch_idx}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Create required files for epoch completion
    with open(os.path.join(epoch_dir, "training_losses.pt"), 'wb') as f:
        torch.save({}, f)
    with open(os.path.join(epoch_dir, "optimizer_buffer.json"), 'w') as f:
        json.dump({}, f)
    with open(os.path.join(epoch_dir, "validation_scores.json"), 'w') as f:
        json.dump({"aggregated": scores}, f)


def test_early_stopping_initialization():
    """Test early stopping initialization with different configurations."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Test enabled configuration
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=5,
            work_dir=work_dir,
            tot_epochs=10,
            metric=TestMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        assert early_stopping.enabled == True
        assert early_stopping.patience == 5
        assert early_stopping.should_stop() == False
        assert len(early_stopping.score_history) == 0

        # Test disabled configuration
        disabled_early_stopping = EarlyStopping(
            enabled=False,
            epochs=10,
            work_dir=work_dir,
            tot_epochs=10,
            metric=TestMetric()
        )

        assert disabled_early_stopping.enabled == False
        assert disabled_early_stopping.should_stop() == False


def test_early_stopping_improvement_detection():
    """Test early stopping detects improvement correctly."""
    with tempfile.TemporaryDirectory() as work_dir:
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=3,  # patience = 3
            work_dir=work_dir,
            tot_epochs=10,
            metric=TestMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        # Create improving scores (lower loss = better for DIRECTION=-1)
        create_epoch_with_scores(work_dir, 0, {"loss": 1.0})
        create_epoch_with_scores(work_dir, 1, {"loss": 0.8})  # improvement
        create_epoch_with_scores(work_dir, 2, {"loss": 0.6})  # improvement

        early_stopping.update()

        assert len(early_stopping.score_history) == 3
        assert early_stopping.epochs_without_improvement == 0  # Still improving
        assert early_stopping.should_stop() == False


def test_early_stopping_triggers_correctly():
    """Test early stopping triggers after patience epochs without improvement."""
    with tempfile.TemporaryDirectory() as work_dir:
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=2,  # patience = 2
            work_dir=work_dir,
            tot_epochs=10,
            metric=TestMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        # Create scores: improvement, then no improvement for 2 epochs
        create_epoch_with_scores(work_dir, 0, {"loss": 1.0})
        create_epoch_with_scores(work_dir, 1, {"loss": 0.5})  # improvement (best)
        create_epoch_with_scores(work_dir, 2, {"loss": 0.7})  # worse (no improvement)
        create_epoch_with_scores(work_dir, 3, {"loss": 0.8})  # worse (no improvement)

        early_stopping.update()

        assert len(early_stopping.score_history) == 4
        assert early_stopping.epochs_without_improvement == 2
        assert early_stopping.should_stop() == True


def test_early_stopping_with_incomparable_vectors():
    """Test early stopping with multi-metric scenarios (vector comparison)."""
    class MultiMetric(SingleTaskMetric):
        DIRECTIONS = {"loss1": -1, "loss2": -1}
        def _compute_score(self, y_pred, y_true):
            return {"loss1": torch.tensor(0.5), "loss2": torch.tensor(0.3)}

    with tempfile.TemporaryDirectory() as work_dir:
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=2,
            work_dir=work_dir,
            tot_epochs=10,
            metric=MultiMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        # Create incomparable scores: [1.0, 0.5] vs [0.8, 0.7] (incomparable in partial order)
        create_epoch_with_scores(work_dir, 0, {"loss1": 1.0, "loss2": 0.5})
        create_epoch_with_scores(work_dir, 1, {"loss1": 0.8, "loss2": 0.7})  # incomparable (treated as no improvement)
        create_epoch_with_scores(work_dir, 2, {"loss1": 0.9, "loss2": 0.8})  # incomparable (treated as no improvement)

        early_stopping.update()

        assert early_stopping.epochs_without_improvement == 2
        assert early_stopping.should_stop() == True


def test_early_stopping_consecutive_epochs():
    """Test early stopping handles non-consecutive epochs correctly."""
    with tempfile.TemporaryDirectory() as work_dir:
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=2,
            work_dir=work_dir,
            tot_epochs=10,
            metric=TestMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        # Create epochs 0, 1, 3 (missing epoch 2)
        create_epoch_with_scores(work_dir, 0, {"loss": 1.0})
        create_epoch_with_scores(work_dir, 1, {"loss": 0.8})
        create_epoch_with_scores(work_dir, 3, {"loss": 0.6})  # Should not be read due to missing epoch 2

        early_stopping.update()

        # Should only read epochs 0, 1 (stops at missing epoch 2)
        assert len(early_stopping.score_history) == 2
        assert early_stopping.last_read_epoch == 1


def test_early_stopping_was_triggered_at_epoch():
    """Test detection of early stopping in completed runs."""
    with tempfile.TemporaryDirectory() as work_dir:
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=2,  # patience = 2
            work_dir=work_dir,
            tot_epochs=10,
            metric=TestMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        # Create a scenario where early stopping should trigger at epoch 3
        create_epoch_with_scores(work_dir, 0, {"loss": 1.0})
        create_epoch_with_scores(work_dir, 1, {"loss": 0.5})  # best
        create_epoch_with_scores(work_dir, 2, {"loss": 0.7})  # no improvement (1st)
        create_epoch_with_scores(work_dir, 3, {"loss": 0.8})  # no improvement (2nd) -> should trigger

        # Test different epochs
        assert early_stopping.was_triggered_at_epoch(0) == False  # Too early
        assert early_stopping.was_triggered_at_epoch(1) == False  # Too early
        assert early_stopping.was_triggered_at_epoch(2) == False  # Only 1 epoch without improvement
        assert early_stopping.was_triggered_at_epoch(3) == True   # 2 epochs without improvement


@pytest.mark.parametrize("direction,improving_scores,worsening_scores", [
    (-1, [1.0, 0.8, 0.6], [0.6, 0.8, 1.0]),  # Loss metric (lower better)
    (1, [0.6, 0.8, 1.0], [1.0, 0.8, 0.6]),   # Accuracy metric (higher better)
])
def test_early_stopping_direction_handling(direction, improving_scores, worsening_scores):
    """Test early stopping works correctly with different metric directions."""
    class DirectionalMetric(SingleTaskMetric):
        DIRECTIONS = {"score": direction}
        def _compute_score(self, y_pred, y_true):
            return {"score": torch.tensor(0.5)}

    with tempfile.TemporaryDirectory() as work_dir:
        early_stopping = EarlyStopping(
            enabled=True,
            epochs=2,
            work_dir=work_dir,
            tot_epochs=10,
            metric=DirectionalMetric(),
            expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        )

        # Test improving scores - should not trigger early stopping
        for i, score in enumerate(improving_scores):
            create_epoch_with_scores(work_dir, i, {"score": score})

        early_stopping.update()
        assert early_stopping.should_stop() == False

        # Reset and test worsening scores - should trigger early stopping
        early_stopping.score_history = []
        early_stopping.best_scores = None
        early_stopping.epochs_without_improvement = 0
        early_stopping.should_stop_early = False
        early_stopping.last_read_epoch = -1

        # Clear work directory and create worsening scores
        for i in range(len(improving_scores)):
            epoch_dir = os.path.join(work_dir, f"epoch_{i}")
            if os.path.exists(epoch_dir):
                import shutil
                shutil.rmtree(epoch_dir)

        for i, score in enumerate(worsening_scores):
            create_epoch_with_scores(work_dir, i, {"score": score})

        early_stopping.update()
        assert early_stopping.should_stop() == True


def test_early_stopping_edge_cases():
    """Test early stopping edge cases and error conditions."""
    # Test with None metric
    with pytest.raises(AssertionError, match="Metric cannot be None"):
        from runners.model_comparison import get_metric_directions
        get_metric_directions(None)

    # Test with metric missing DIRECTIONS
    class InvalidMetric:
        pass

    with pytest.raises(AttributeError, match="has no DIRECTIONS attribute"):
        from runners.model_comparison import get_metric_directions
        get_metric_directions(InvalidMetric())


def test_early_stopping_deterministic():
    """Test early stopping produces consistent results with same input."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create deterministic scores
        scores = [{"loss": 1.0}, {"loss": 0.8}, {"loss": 0.9}, {"loss": 1.1}]

        for epoch_idx, score in enumerate(scores):
            create_epoch_with_scores(work_dir, epoch_idx, score)

        # Run early stopping twice with same configuration
        results = []
        for _ in range(2):
            early_stopping = EarlyStopping(
                enabled=True,
                epochs=2,
                work_dir=work_dir,
                tot_epochs=10,
                metric=TestMetric(),
                expected_files=["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
            )
            early_stopping.update()
            results.append({
                'should_stop': early_stopping.should_stop(),
                'epochs_without_improvement': early_stopping.epochs_without_improvement,
                'history_length': len(early_stopping.score_history)
            })

        # Results should be identical
        assert results[0] == results[1]
