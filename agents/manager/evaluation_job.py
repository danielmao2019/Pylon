from __future__ import annotations

from typing import List
import os
import json

from agents.manager.progress_info import ProgressInfo
from agents.manager.base_job import BaseJob


class EvaluationJob(BaseJob):
    """Copied evaluator logic as classmethods for manager jobs."""

    EXPECTED_FILES = ["evaluation_scores.json"]
    LOG_PATTERN = "eval_*.log"

    # ====================================================================================================
    # 
    # ====================================================================================================

    def get_progress(self, force_progress_recompute: bool = False) -> ProgressInfo:
        eval_complete = self._check_files_exist(self.work_dir)

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
        for filename in cls.EXPECTED_FILES:
            filepath = os.path.join(work_dir, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return False
            try:
                with open(filepath, 'r') as f:
                json.load(f)
        except (json.JSONDecodeError, IOError):
            return False
        return True

    # ====================================================================================================
    # 
    # ====================================================================================================

    def get_log_last_update(self) -> Optional[float]:
        return

    def get_epoch_last_update(self) -> Optional[float]:
        return
