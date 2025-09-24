from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Literal, Optional
from abc import ABC, abstractmethod

from agents.monitor.process_info import ProcessInfo
from agents.manager.progress_info import ProgressInfo
from utils.io.config import load_config


_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class BaseJob(ABC):
    """Object-oriented representation of a single training job."""

    def __init__(self, config_filepath: str) -> None:
        self.config_filepath = config_filepath
        self.work_dir: Optional[str] = None
        self.progress: Optional[ProgressInfo] = None
        self.status: Optional[_JobStatus] = None
        self.process_info: Optional[ProcessInfo] = None

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
        work_dir = self.get_work_dir(self.config_filepath)
        config_dict = load_config(self.config_filepath)

        # Use polymorphism: subclass provides its own behavior
        progress = self.calculate_progress(work_dir, config_dict, force_progress_recompute=force_progress_recompute)
        self.status = self.get_status()
        self.work_dir = work_dir
        self.progress = progress
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

    @classmethod
    @abstractmethod
    def calculate_progress(
        cls,
        work_dir: str,
        config: Optional[Dict],
        *,
        force_progress_recompute: bool = False,
    ) -> ProgressInfo:
        ...

    # ====================================================================================================
    #
    # ====================================================================================================

    def get_status():
        log_pattern = self.get_log_pattern()
        expected_files = self.get_expected_files()

        log_last_update = self.get_log_last_update(work_dir, log_pattern)
        epoch_last_update = self.get_epoch_last_update(work_dir, expected_files)

        is_running_status = (
            log_last_update is not None and (time.time() - log_last_update <= sleep_time)
        )
        is_complete = progress.completed_epochs >= epochs

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

    @classmethod
    @abstractmethod
    def get_log_pattern(cls) -> str:
        ...

    @classmethod
    @abstractmethod
    def get_expected_files(cls) -> List[str]:
        ...

    @staticmethod
    def get_log_last_update(
        work_dir: str,
        log_pattern: str = 'train_val*.log',
    ) -> Optional[float]:
        """Get the timestamp of the last log update."""
        if not os.path.isdir(work_dir):
            return None
        logs = glob.glob(os.path.join(work_dir, log_pattern))
        if not logs:
            return None
        return max(os.path.getmtime(fp) for fp in logs)

    @staticmethod
    def get_epoch_last_update(
        work_dir: str,
        expected_files: List[str],
    ) -> Optional[float]:
        """Get the timestamp of the last epoch file update."""
        if not os.path.isdir(work_dir):
            return None
        epoch_dirs = glob.glob(os.path.join(work_dir, 'epoch_*'))
        if not epoch_dirs:
            return None
        timestamps = [
            os.path.getmtime(os.path.join(epoch_dir, filename))
            for epoch_dir in epoch_dirs
            for filename in expected_files
            if os.path.isfile(os.path.join(epoch_dir, filename))
        ]
        return max(timestamps) if timestamps else None

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
