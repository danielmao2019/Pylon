from __future__ import annotations

import os
import time
from typing import Dict, Literal, Optional
from abc import ABC, abstractmethod

from agents.monitor.process_info import ProcessInfo
from agents.manager.progress_info import ProgressInfo
from utils.io.config import load_config


_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class BaseJob(ABC):
    """Object-oriented representation of a single training job."""

    def __init__(self, config_filepath: str) -> None:
        self.config_filepath = config_filepath
        self.work_dir: str = self.get_work_dir(self.config_filepath)
        self.config_dict = load_config(self.config_filepath)
        self.progress: ProgressInfo
        self.status: _JobStatus
        self.process_info: ProcessInfo

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
        self.progress = self.get_progress(force_progress_recompute)
        self.status = self.get_status(sleep_time, epochs, outdated_days, config_to_process_info)
        self.process_info = config_to_process_info.get(self.config_filepath)

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
        ...

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
        epoch_last_update = self.get_epoch_last_update()

        is_running_status = (
            log_last_update is not None and (time.time() - log_last_update <= sleep_time)
        )
        is_complete = self.progress.completed_epochs >= epochs

        if is_running_status:
            status: _JobStatus = 'running'
        elif is_complete:
            if epoch_last_update is not None and (
                time.time() - epoch_last_update > outdated_days * 24 * 60 * 60
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
        ...

    @abstractmethod
    def get_epoch_last_update(self) -> Optional[float]:
        ...

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
