from __future__ import annotations

import glob
import os
import time
from abc import ABC
from typing import Optional

from agents.manager.base_job import BaseJob
from utils.io.config import load_config
from agents.manager.job_types import RunnerKind
from agents.manager.runtime import JobRuntimeParams


class DefaultJob(BaseJob, ABC):
    """Object-oriented representation of a single training job."""

    runner_kind: RunnerKind | None = None
    LOG_PATTERN: str = ""

    def __init__(self, command: str) -> None:
        config_filepath = self.parse_config(command)
        self.config_filepath = config_filepath
        self.config_dict = load_config(self.config_filepath)
        runner_kind = getattr(self.__class__, "runner_kind", None)
        if runner_kind is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define runner_kind"
            )
        self.runner_kind: RunnerKind = runner_kind
        super().__init__(command=command)

    def derive_work_dir(self) -> str:
        config_path = self.parse_config(self.command)
        rel_path = os.path.splitext(os.path.relpath(config_path, start='./configs'))[0]
        return os.path.join('./logs', rel_path)

    @staticmethod
    def parse_config(cmd: str) -> str:
        """Extract config filepath from command string."""
        assert isinstance(cmd, str), f"cmd={cmd!r}"
        assert 'python' in cmd, f"cmd={cmd!r}"
        assert '--config-filepath' in cmd, f"cmd={cmd!r}"
        parts = cmd.split(' ')
        for idx, part in enumerate(parts):
            if part == '--config-filepath':
                return parts[idx + 1]
        raise AssertionError('Config filepath not found in command string')

    # ------------------------------------------------------------------
    # compute status helpers
    # ------------------------------------------------------------------

    def is_active(self, runtime: JobRuntimeParams) -> bool:
        last_log = self.get_log_last_update()
        if last_log is None:
            return False
        return (time.time() - last_log) <= runtime.sleep_time

    def is_stuck(self, runtime: JobRuntimeParams) -> bool:
        return self.command in runtime.command_processes

    def get_log_last_update(self) -> Optional[float]:
        assert self.LOG_PATTERN, (
            "Cannot call get_log_last_update on DefaultJob without defining LOG_PATTERN."
        )
        if not os.path.isdir(self.work_dir):
            return None
        logs = glob.glob(os.path.join(self.work_dir, self.LOG_PATTERN))
        if not logs:
            return None
        return max(os.path.getmtime(filepath) for filepath in logs)

    def tmux_session_name(self) -> str:
        """Return a tmux-friendly identifier relative to ``./configs/benchmarks``."""

        config_path = os.path.abspath(self.config_filepath)
        benchmarks_root = os.path.abspath(os.path.join('.', 'configs', 'benchmarks'))

        relative_path = os.path.relpath(config_path, benchmarks_root)
        assert not relative_path.startswith('..') and relative_path != '.', (
            f"Config {self.config_filepath!r} must live under './configs/benchmarks'"
        )

        return relative_path
