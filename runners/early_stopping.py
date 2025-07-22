from typing import Dict, Any, Optional, List
import os
import json
from runners.model_comparison import compare_scores, get_metric_directions
from utils.automation.progress_tracking.session_progress import check_epoch_finished


class EarlyStopping:
    """
    Early stopping utility that tracks validation scores and determines when to stop training.

    Maintains internal score history and avoids redundant I/O operations.
    """

    def __init__(
        self,
        enabled: bool = True,
        epochs: int = 10,
        work_dir: str = None,
        tot_epochs: int = None,
        metric = None,
        expected_files: List[str] = None,
        logger = None
    ):
        """
        Initialize early stopping.

        Args:
            enabled: Whether early stopping is enabled
            epochs: Patience epochs without improvement (patience)
            work_dir: Working directory where epoch results are stored
            tot_epochs: Total number of training epochs
            metric: Metric object for extracting DIRECTION attributes
            expected_files: List of expected files for epoch completion check
            logger: Optional logger for info messages
        """
        self.enabled = enabled
        self.patience = epochs
        self.work_dir = work_dir
        self.tot_epochs = tot_epochs
        self.metric = metric
        self.expected_files = expected_files or []
        self.logger = logger

        # Internal state
        self.score_history: List[Dict[str, Any]] = []
        self.best_scores: Optional[Dict[str, Any]] = None
        self.epochs_without_improvement = 0
        self.should_stop_early = False
        self.last_read_epoch = -1  # Track last epoch we've read to avoid redundant I/O

    def update(self) -> None:
        """
        Update score history by reading new epoch results.

        Uses a while loop to read all available epochs since last update.
        Avoids redundant I/O by tracking the last read epoch.
        """
        if not self.enabled or self.work_dir is None:
            return

        # Read scores from all completed epochs we haven't read yet
        current_epoch = self.last_read_epoch + 1

        while current_epoch < self.tot_epochs:
            epoch_dir = os.path.join(self.work_dir, f"epoch_{current_epoch}")

            # Check if epoch is completed
            if not check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=self.expected_files
            ):
                break

            # Read validation scores (check_epoch_finished already verified file exists)
            scores_path = os.path.join(epoch_dir, "validation_scores.json")
            with open(scores_path, 'r') as f:
                validation_scores = json.load(f)

            # Extract aggregated scores
            aggregated_scores = validation_scores.get('aggregated', {})
            assert aggregated_scores, f"Missing 'aggregated' scores in {scores_path} - this should not happen"

            self.score_history.append(aggregated_scores)
            self._update_early_stopping_state(aggregated_scores)
            self.last_read_epoch = current_epoch
            current_epoch += 1

    def _update_early_stopping_state(self, current_scores: Dict[str, Any]) -> None:
        """Update early stopping state with new scores."""
        # Get metric directions
        metric_directions = get_metric_directions(self.metric)
        if not metric_directions:
            return

        # Check if this is the first epoch or if we have improvement
        if self.best_scores is None:
            # First epoch
            self.best_scores = current_scores.copy()
            self.epochs_without_improvement = 0
            return

        # Compare current scores with best scores using vector comparison
        is_better = compare_scores(
            current_scores=current_scores,
            best_scores=self.best_scores,
            order_config=False,  # Vector comparison
            metric_directions=metric_directions
        )

        if is_better:
            # Found improvement
            self.best_scores = current_scores.copy()
            self.epochs_without_improvement = 0
        else:
            # No improvement (including incomparable cases)
            self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                self.should_stop_early = True
                if self.logger:
                    self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")

    def should_stop(self) -> bool:
        """
        Check if training should stop early.

        Returns:
            True if training should stop early
        """
        return self.enabled and self.should_stop_early

    def was_triggered_at_epoch(self, epoch_idx: int) -> bool:
        """
        Check if early stopping would have been triggered at the given epoch.

        Used for detecting early stopping completion during resumption.

        Args:
            epoch_idx: Epoch index to check

        Returns:
            True if early stopping was triggered at this epoch
        """
        if not self.enabled or epoch_idx < self.patience:
            return False

        # Simulate early stopping check with historical data
        # Read score history for the required epochs
        validation_scores_history = []
        for i in range(max(0, epoch_idx - self.patience), epoch_idx + 1):
            epoch_dir = os.path.join(self.work_dir, f"epoch_{i}")
            scores_path = os.path.join(epoch_dir, "validation_scores.json")

            # Check if epoch is complete (includes file existence check)
            if not check_epoch_finished(
                epoch_dir=epoch_dir,
                expected_files=self.expected_files
            ):
                # Epochs must be consecutive - if any is incomplete, we can't determine early stopping
                return False

            with open(scores_path, 'r') as f:
                scores = json.load(f)
                aggregated = scores.get('aggregated', {})
                assert aggregated, f"Missing 'aggregated' scores in {scores_path} - this should not happen"
                validation_scores_history.append(aggregated)

        # Check if there was no improvement for 'patience' epochs
        if len(validation_scores_history) >= self.patience + 1:
            return self._would_early_stop_with_history(validation_scores_history, self.patience)

        return False

    def _would_early_stop_with_history(self, score_history: List[Dict], patience: int) -> bool:
        """Check if early stopping would have been triggered given the score history."""
        if len(score_history) < patience + 1:
            return False

        # Get metric directions
        metric_directions = get_metric_directions(self.metric)
        if not metric_directions:
            return False

        # Check if the last 'patience' epochs had no improvement
        best_scores = score_history[0]
        epochs_without_improvement = 0

        for i, current_scores in enumerate(score_history[1:], 1):
            is_better = compare_scores(
                current_scores=current_scores,
                best_scores=best_scores,
                order_config=False,  # Vector comparison
                metric_directions=metric_directions
            )

            if is_better:
                best_scores = current_scores
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        return epochs_without_improvement >= patience
