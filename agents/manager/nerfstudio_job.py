from __future__ import annotations

import json
import os
import shlex
from typing import Dict, Optional

from agents.manager.base_job import BaseJob
from agents.manager.progress_info import ProgressInfo


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    def __init__(
        self,
        command: str,
    ) -> None:
        self._work_dir = self._resolve_work_dir(command)
        self._expected_steps = self._extract_int_flag(command)
        self._last_metrics: Optional[Dict[str, object]] = None
        super().__init__(command=command)

    def compute_progress(self) -> ProgressInfo:
        data = self._load_metrics()
        self._last_metrics = data

        completed_steps = int(data.get("latest_step", data.get("completed_steps", 0)) or 0)
        total_steps = data.get("total_steps", self._expected_steps)
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

    def compute_status(self, progress: ProgressInfo) -> str:
        metrics = self._last_metrics or self._load_metrics()
        self._last_metrics = metrics

        expected_total = metrics.get("total_steps")
        if expected_total is None:
            expected_total = self._expected_steps if self._expected_steps is not None else progress.total_epochs
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
        if self._did_fail(metrics):
            return "failed"
        return "pending"

    def derive_work_dir(self) -> str:
        return self._work_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_metrics(self) -> Dict[str, object]:
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

    @staticmethod
    def _resolve_work_dir(command: str) -> str:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        lookup_flags = (
            '--output-dir', '--output_dir', '--output',
            '--workspace', '--run-dir', '--run_dir', '--logdir', '--log-dir'
        )
        for flag in lookup_flags:
            if flag in tokens:
                idx = tokens.index(flag)
                if idx + 1 < len(tokens):
                    candidate = tokens[idx + 1]
                    if candidate:
                        return os.path.normpath(os.path.expanduser(candidate))
            prefix = f"{flag}="
            for token in tokens:
                if token.startswith(prefix) and len(token) > len(prefix):
                    return os.path.normpath(os.path.expanduser(token[len(prefix):]))

        raise ValueError(
            "Unable to determine work directory for NerfStudioJob. "
            "Provide a supported command flag (e.g. --output-dir) in the command."
        )

    @staticmethod
    def _extract_int_flag(command: str) -> Optional[int]:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        candidate_flags = (
            '--max-num-iterations', '--max_num_iterations',
            '--steps', '--total-steps', '--total_steps'
        )
        for flag in candidate_flags:
            if flag in tokens:
                idx = tokens.index(flag)
                if idx + 1 < len(tokens):
                    value = tokens[idx + 1]
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            prefix = f"{flag}="
            for token in tokens:
                if token.startswith(prefix):
                    try:
                        return int(token[len(prefix):])
                    except (TypeError, ValueError):
                        continue
        return None

    @staticmethod
    def _did_fail(metrics: Dict[str, object]) -> bool:
        flag = metrics.get("had_error")
        if isinstance(flag, bool):
            return flag
        status = metrics.get("status")
        if isinstance(status, str) and status.lower() in {"failed", "error", "crashed"}:
            return True
        errors = metrics.get("errors")
        if isinstance(errors, (list, tuple)) and len(errors) > 0:
            return True
        return False
