from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from agents.manager.progress_info import ProgressInfo
from agents.manager.base_job import BaseJob
from utils.io.config import load_config
from agents.manager.job_types import RunnerKind
from agents.manager.runtime import JobRuntimeParams


_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class DefaultJob(BaseJob, ABC):
    """Object-oriented representation of a single training job."""

    runner_kind: RunnerKind | None = None

    def __init__(self, config_filepath: str) -> None:
        self.config_filepath = config_filepath
        self.config_dict = load_config(self.config_filepath)
        runner_kind = getattr(self.__class__, "runner_kind", None)
        if runner_kind is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define runner_kind"
            )
        self.runner_kind: RunnerKind = runner_kind
        command = f"python main.py --config-filepath {config_filepath}"
        super().__init__(command=command)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_active(self, runtime: JobRuntimeParams) -> bool:
        last_log = self.get_log_last_update()
        if last_log is None:
            return False
        return (time.time() - last_log) <= runtime.sleep_time

    @abstractmethod
    def is_complete(self, progress: ProgressInfo, runtime: JobRuntimeParams) -> bool:
        """Return True when the job should count as complete."""

    def is_stuck(self, runtime: JobRuntimeParams) -> bool:
        return self.command in runtime.command_processes

    # ====================================================================================================
    # 
    # ====================================================================================================

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': self.config_filepath,
            'work_dir': self.work_dir,
            'progress': self._serialize(self.progress),
            'status': self.status,
            'process_info': self._serialize(self.process_info),
        }

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

    @staticmethod
    def _serialize(value):
        if value is None:
            return None
        if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            return value.to_dict()
        return value

    def derive_work_dir(self) -> str:
        rel_path = os.path.splitext(os.path.relpath(self.config_filepath, start='./configs'))[0]
        return os.path.join('./logs', rel_path)

    # ------------------------------------------------------------------
    # Abstract data accessors
    # ------------------------------------------------------------------

    @abstractmethod
    def get_log_last_update(self) -> Optional[float]:
        pass
