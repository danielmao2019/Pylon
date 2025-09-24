from __future__ import annotations

from typing import List
import os
import json

from agents.manager.progress_info import ProgressInfo


class EvaluationJob:
    """Copied evaluator logic as classmethods for manager jobs."""

    @classmethod
    def get_expected_files(cls) -> List[str]:
        return ["evaluation_scores.json"]

    @classmethod
    def get_log_pattern(cls) -> str:
        return "eval_*.log"

    @classmethod
    def calculate_progress(cls, work_dir: str, config: dict | None, force_progress_recompute: bool = False) -> ProgressInfo:
        eval_complete = cls._check_files_exist(work_dir)

        progress_percentage = 100.0 if eval_complete else 0.0
        completed_epochs = 1 if eval_complete else 0

        return ProgressInfo(
            completed_epochs=completed_epochs,
            progress_percentage=progress_percentage,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='evaluator',
            total_epochs=1,
        )

    @classmethod
    def _check_files_exist(cls, work_dir: str) -> bool:
        filepath = os.path.join(work_dir, "evaluation_scores.json")
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        try:
            with open(filepath, 'r') as f:
                json.load(f)
        except (json.JSONDecodeError, IOError):
            return False
        return True

