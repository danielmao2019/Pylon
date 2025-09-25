from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from agents.manager.progress_info import ProgressInfo
from agents.manager.base_job import BaseJob
from agents.monitor.process_info import ProcessInfo
from utils.io.config import load_config


_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class DefaultJob(BaseJob, ABC):
    """Object-oriented representation of a single training job."""

    def __init__(self, config_filepath: str) -> None:
        work_dir = self.get_work_dir(config_filepath)
        command = f"python main.py --config-filepath {config_filepath}"
        super().__init__(command=command, work_dir=work_dir)
        self.config_filepath = config_filepath
        self.config_dict = load_config(self.config_filepath)

    # ====================================================================================================
    # 
    # ====================================================================================================

    def populate(
        self,
        *,
        epochs: int,
        config_to_process_info: Dict[str, ProcessInfo],
        sleep_time: int,
        outdated_days: int,
        force_progress_recompute: bool,
    ) -> None:
        context: Dict[str, Any] = {
            "epochs": epochs,
            "sleep_time": sleep_time,
            "outdated_days": outdated_days,
            "config_to_process_info": config_to_process_info,
            "force_progress_recompute": force_progress_recompute,
        }
        self.attach_process(config_to_process_info.get(self.config_filepath))
        self.refresh(context=context)

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

    @abstractmethod
    def get_progress(
        self,
        force_progress_recompute: bool = False,
    ) -> ProgressInfo:
        pass

    # ====================================================================================================
    #
    # ====================================================================================================


    def get_status(
        self,
        sleep_time: int,
        epochs: int,
        outdated_days: int,
        config_to_process_info: Dict[str, ProcessInfo],
    ) -> _JobStatus:
        log_last_update = self.get_log_last_update()
        artifact_last_update = self.get_artifact_last_update()

        is_running_status = (
            log_last_update is not None and (time.time() - log_last_update <= sleep_time)
        )

        if self.__class__.__name__ == 'EvaluationJob':
            is_complete = self.progress.completed_epochs >= 1
        else:
            is_complete = (
                self.progress.early_stopped
                or self.progress.completed_epochs >= epochs
            )

        if is_running_status:
            status: _JobStatus = 'running'
        elif is_complete:
            if artifact_last_update is not None and (
                time.time() - artifact_last_update > outdated_days * 24 * 60 * 60
            ):
                status = 'outdated'
            else:
                status = 'finished'
        elif self.config_filepath in config_to_process_info:
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

    # ------------------------------------------------------------------
    # BaseJob integration helpers
    # ------------------------------------------------------------------

    def compute_progress(
        self, *, context: Optional[Dict[str, Any]] = None
    ) -> ProgressInfo:
        force = bool((context or {}).get("force_progress_recompute", False))
        return self.get_progress(force_progress_recompute=force)

    def compute_status(
        self,
        *,
        progress: ProgressInfo,
        context: Optional[Dict[str, Any]] = None,
    ) -> _JobStatus:
        ctx = context or {}
        epochs_candidate = ctx.get("epochs", progress.total_epochs or self.config_dict.get('epochs'))
        try:
            epochs_int = int(epochs_candidate) if epochs_candidate is not None else 0
        except (TypeError, ValueError):
            epochs_int = 0

        sleep_time_candidate = ctx.get("sleep_time", 0)
        outdated_candidate = ctx.get("outdated_days", 0)
        try:
            sleep_time_int = int(sleep_time_candidate)
        except (TypeError, ValueError):
            sleep_time_int = 0
        try:
            outdated_days_int = int(outdated_candidate)
        except (TypeError, ValueError):
            outdated_days_int = 0

        config_to_process_info = ctx.get("config_to_process_info", {})
        return self.get_status(
            sleep_time=sleep_time_int,
            epochs=epochs_int,
            outdated_days=outdated_days_int,
            config_to_process_info=config_to_process_info,
        )
