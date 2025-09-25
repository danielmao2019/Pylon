from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from agents.manager.base_job import BaseJob
from agents.manager.progress_info import ProgressInfo


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    def __init__(
        self,
        command: str,
        *,
        work_dir: str,
        env: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(command=command, work_dir=work_dir, env=env, metadata=metadata)

    def _load_metrics(self) -> Dict[str, Any]:
        if not self.work_dir:
            return {}
        metrics_path = os.path.join(self.work_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            return {}
        try:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    def compute_progress(self, *, force: Optional[bool] = None) -> ProgressInfo:
        data = self._load_metrics()

        completed_steps = int(data.get("latest_step", data.get("completed_steps", 0)) or 0)
        total_steps = data.get("total_steps", self.metadata.get("total_steps"))
        try:
            total_steps_int = int(total_steps) if total_steps is not None else None
        except (TypeError, ValueError):
            total_steps_int = None

        if total_steps_int and total_steps_int > 0:
            progress_pct = min(100.0, (completed_steps / total_steps_int) * 100.0)
        elif completed_steps > 0:
            progress_pct = 100.0
        else:
            progress_pct = 0.0

        return ProgressInfo(
            completed_epochs=completed_steps,
            progress_percentage=progress_pct,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='nerfstudio',
            total_epochs=total_steps_int,
        )

    def compute_status(
        self,
        *,
        progress: ProgressInfo,
    ) -> str:
        expected_total = self.metadata.get("total_steps", progress.total_epochs)
        try:
            expected_total_int = int(expected_total) if expected_total is not None else None
        except (TypeError, ValueError):
            expected_total_int = None

        if expected_total_int is not None and expected_total_int > 0:
            if progress.completed_epochs >= expected_total_int:
                return "finished"
        if self.process_info is not None:
            return "running"
        if progress.completed_epochs > 0:
            return "running"
        if self.metadata.get("had_error"):
            return "failed"
        return "pending"
