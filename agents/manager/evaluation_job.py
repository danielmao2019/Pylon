from __future__ import annotations

import json
import os

from agents.manager.progress_info import ProgressInfo
from agents.manager.default_job import DefaultJob
from agents.manager.job_types import RunnerKind
from agents.manager.runtime import JobRuntimeParams


class EvaluationJob(DefaultJob):
    """Copied evaluator logic as classmethods for manager jobs."""

    EXPECTED_FILES = ["evaluation_scores.json"]
    LOG_PATTERN = "eval_*.log"
    runner_kind = RunnerKind.EVALUATOR

    # ====================================================================================================
    # 
    # ====================================================================================================

    def compute_progress(self, runtime: JobRuntimeParams) -> ProgressInfo:
        eval_complete = True
        for filename in self.EXPECTED_FILES:
            filepath = os.path.join(self.work_dir, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                eval_complete = False
                break
            try:
                with open(filepath, 'r', encoding='utf-8') as file_obj:
                    json.load(file_obj)
            except (json.JSONDecodeError, OSError):
                eval_complete = False
                break

        progress_percentage = 100.0 if eval_complete else 0.0
        completed_epochs = 1 if eval_complete else 0

        return ProgressInfo(
            completed_epochs=completed_epochs,
            progress_percentage=progress_percentage,
            early_stopped=False,
            early_stopped_at_epoch=None,
            total_epochs=1,
        )

    # ====================================================================================================
    # 
    # ====================================================================================================

    def is_complete(
        self,
        progress: ProgressInfo,
        runtime: JobRuntimeParams,
    ) -> bool:
        return progress.completed_epochs >= 1
