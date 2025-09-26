from __future__ import annotations

import json
import os
import shlex
from typing import Dict, Optional

from agents.manager.base_job import BaseJob
from agents.manager.progress_info import ProgressInfo


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    def __init__(self, command: str) -> None:
        self._expected_steps = self._extract_int_flag(command)
        super().__init__(command=command)

    def compute_progress(self) -> ProgressInfo:
        data = self._load_metrics()

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

    def derive_work_dir(self) -> str:
        try:
            tokens = shlex.split(self.command)
        except ValueError:
            tokens = self.command.split()

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

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        return self.process_info is not None

    def is_complete(self, progress: ProgressInfo) -> bool:
        target = progress.total_epochs or self._expected_steps
        try:
            target_int = int(target) if target is not None else 0
        except (TypeError, ValueError):
            target_int = 0
        if target_int <= 0:
            return False
        return progress.completed_epochs >= target_int

    def is_outdated(self) -> bool:
        return False

    def is_stuck(self) -> bool:
        return False

    def is_failed(self, progress: ProgressInfo) -> bool:
        return not self.is_complete(progress)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_metrics(self) -> Dict[str, object]:
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
