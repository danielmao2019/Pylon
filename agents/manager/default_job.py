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
        default_epochs = int(self.config_dict.get('epochs', 0) or 0)
        self._runtime: JobRuntimeParams = JobRuntimeParams(
            epochs=default_epochs,
            sleep_time=0,
            outdated_days=0,
            config_processes={},
            force_progress_recompute=False,
        )

    # ====================================================================================================
    # 
    # ====================================================================================================

    def configure(self, runtime: JobRuntimeParams) -> None:
        """Attach runtime parameters and recompute state."""
        self._runtime = runtime
        self.attach_process(runtime.process_for(self.config_filepath))
        progress = self.compute_progress()
        self.progress = progress
        self.status = self.compute_status(progress)

    # ====================================================================================================
    # 
    # ====================================================================================================

    def compute_status(self, progress: ProgressInfo) -> _JobStatus:
        runtime = self.runtime
        now = time.time()

        if self._is_active(runtime, now):
            return 'running'

        if self._is_complete(progress, runtime):
            return 'outdated' if self._is_outdated(runtime, now) else 'finished'

        if self._is_stuck(runtime):
            return 'stuck'

        return 'failed'

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

    @property
    def runtime(self) -> JobRuntimeParams:
        return self._runtime

    def derive_work_dir(self) -> str:
        rel_path = os.path.splitext(os.path.relpath(self.config_filepath, start='./configs'))[0]
        return os.path.join('./logs', rel_path)

    # ------------------------------------------------------------------
    # Status helpers (overridable by subclasses)
    # ------------------------------------------------------------------

    def _is_active(self, runtime: JobRuntimeParams, now: float) -> bool:
        last_log = self.get_log_last_update()
        if last_log is None:
            return False
        return (now - last_log) <= runtime.sleep_time

    @abstractmethod
    def _is_complete(
        self,
        progress: ProgressInfo,
        runtime: JobRuntimeParams,
    ) -> bool:
        """Return True when the job should count as complete."""

    def _is_outdated(self, runtime: JobRuntimeParams, now: float) -> bool:
        if runtime.outdated_days <= 0:
            return False
        artifact_last_update = self.get_artifact_last_update()
        if artifact_last_update is None:
            return False
        stale_after = runtime.outdated_days * 24 * 60 * 60
        return (now - artifact_last_update) > stale_after

    def _is_stuck(self, runtime: JobRuntimeParams) -> bool:
        return self.config_filepath in runtime.config_processes

    # ------------------------------------------------------------------
    # Abstract data accessors
    # ------------------------------------------------------------------

    @abstractmethod
    def get_log_last_update(self) -> Optional[float]:
        pass

    @abstractmethod
    def get_artifact_last_update(self) -> Optional[float]:
        pass
