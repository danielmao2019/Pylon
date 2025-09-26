from __future__ import annotations

import json
import os
import shlex
from typing import Dict, Optional, Sequence

from agents.manager.base_job import BaseJob
from agents.manager.progress_info import ProgressInfo


class NerfStudioJob(BaseJob):
    """Minimal job wrapper for NeRFStudio commands."""

    def __init__(self, command: str) -> None:
        self._expected_steps = self._extract_int_flag(command)
        super().__init__(command=command)

    def compute_progress(self) -> ProgressInfo:
        metrics = self._load_metrics()

        completed_steps = self._extract_int(metrics, (
            "latest_step",
            "step",
            "completed_steps",
            "iteration",
            "global_step",
        ))
        assert completed_steps is not None, (
            "NerfStudioJob expects metrics.json to include a step counter"
        )

        total_steps = self._extract_int(metrics, (
            "total_steps",
            "max_num_iterations",
            "max_iterations",
            "num_iterations",
        ))
        if total_steps is None:
            total_steps = self._expected_steps
        assert total_steps is not None, (
            "Provide --max-num-iterations or ensure metrics.json exposes total_steps"
        )
        assert total_steps > 0, "Total steps must be positive"

        progress_pct = min(100.0, (completed_steps / total_steps) * 100.0)

        return ProgressInfo(
            completed_epochs=completed_steps,
            progress_percentage=progress_pct,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='nerfstudio',
            total_epochs=total_steps,
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
        for filename in ("metrics.json", "metrics.jsonl"):
            path = os.path.join(self.work_dir, filename)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    if filename.endswith(".jsonl"):
                        last_payload = None
                        for line in handle:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            last_payload = json.loads(stripped)
                        assert last_payload is not None, f"{filename} is empty"
                        assert isinstance(last_payload, dict), (
                            f"Expected dict objects in {filename}"
                        )
                        return last_payload
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise AssertionError(f"Failed to read {filename}: {exc}") from exc

            assert isinstance(data, dict), f"Expected dict payload in {filename}"
            return data

        raise AssertionError(
            "NerfStudioJob expects metrics.json or metrics.jsonl in the work directory"
        )

    @staticmethod
    def _extract_int(mapping: Dict[str, object], keys: Sequence[str]) -> Optional[int]:
        for key in keys:
            if key not in mapping:
                continue
            value = NerfStudioJob._coerce_non_negative_int(mapping.get(key))
            if value is not None:
                return value
        return None

    @staticmethod
    def _coerce_non_negative_int(value: object) -> Optional[int]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            number = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            try:
                number = int(float(str(value)))
            except (TypeError, ValueError):
                return None
        return number if number >= 0 else None

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
