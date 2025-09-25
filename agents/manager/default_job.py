from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from agents.manager.progress_info import ProgressInfo
from agents.manager.base_job import BaseJob
from agents.monitor.process_info import ProcessInfo
from utils.io.config import load_config
from agents.manager.job_types import RunnerKind
from agents.manager.runtime import JobRuntimeParams


_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class DefaultJob(BaseJob, ABC):
    """Object-oriented representation of a single training job."""

    runner_kind: RunnerKind | None = None

    def __init__(self, config_filepath: str) -> None:
        work_dir = self.get_work_dir(config_filepath)
        command = f"python main.py --config-filepath {config_filepath}"
        super().__init__(command=command, work_dir=work_dir)
        self.config_filepath = config_filepath
        self.config_dict = load_config(self.config_filepath)
        runner_kind = getattr(self.__class__, "runner_kind", None)
        if runner_kind is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define runner_kind"
            )
        self.runner_kind: RunnerKind = runner_kind
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
        self.refresh()

    # ====================================================================================================
    # 
    # ====================================================================================================

    @classmethod
    def get_work_dir(cls, config_filepath: str) -> str:
        """Derive work directory path from a config file path."""
        rel_path = os.path.splitext(os.path.relpath(config_filepath, start='./configs'))[0]
        return os.path.join('./logs', rel_path)

    @classmethod
    def get_config_filepath(cls, work_dir: str) -> str:
        """Derive config file path from a work directory path."""
        rel_path = os.path.relpath(work_dir, './logs')
        return os.path.join('./configs', rel_path) + '.py'

    # ====================================================================================================
    # 
    # ====================================================================================================

    def compute_status(
        self,
        *,
        progress: ProgressInfo,
    ) -> _JobStatus:
        runtime = self.runtime
        log_last_update = self.get_log_last_update()
        artifact_last_update = self.get_artifact_last_update()

        is_running_status = (
            log_last_update is not None and (time.time() - log_last_update <= runtime.sleep_time)
        )

        if self.runner_kind is RunnerKind.EVALUATOR:
            is_complete = progress.completed_epochs >= 1
        else:
            is_complete = (
                progress.early_stopped
                or (
                    runtime.epochs > 0
                    and progress.completed_epochs >= runtime.epochs
                )
            )

        if is_running_status:
            status: _JobStatus = 'running'
        elif is_complete:
            if artifact_last_update is not None and (
                time.time() - artifact_last_update > runtime.outdated_days * 24 * 60 * 60
            ):
                status = 'outdated'
            else:
                status = 'finished'
        elif self.config_filepath in runtime.config_processes:
            status = 'stuck'
        else:
            status = 'failed'
        return status

    @abstractmethod
    def get_log_last_update(self) -> Optional[float]:
        pass

    @abstractmethod
    def get_artifact_last_update(self) -> Optional[float]:
        pass

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

    @property
    def runtime(self) -> JobRuntimeParams:
        return self._runtime
