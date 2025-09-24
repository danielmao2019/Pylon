from __future__ import annotations

from typing import List

from agents.manager.progress_info import ProgressInfo


class TrainingJob:
    """Copied trainer logic as classmethods for manager jobs."""

    @classmethod
    def get_expected_files(cls) -> List[str]:
        return ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]

    @classmethod
    def get_log_pattern(cls) -> str:
        return "train_val*.log"

    @classmethod
    def calculate_progress(cls, work_dir: str, config: dict | None, force_progress_recompute: bool = False) -> ProgressInfo:
        from agents.manager.session_progress import get_session_progress
        basic_progress = get_session_progress(work_dir, cls.get_expected_files(), force_progress_recompute=force_progress_recompute)

        total_epochs = config.get('epochs') if isinstance(config, dict) else None

        return ProgressInfo(
            completed_epochs=basic_progress.completed_epochs,
            progress_percentage=basic_progress.progress_percentage,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='trainer',
            total_epochs=total_epochs,
        )
