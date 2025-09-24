from __future__ import annotations

import glob
import os
import time
from typing import Dict, List, Literal, Optional

from agents.monitor.process_info import ProcessInfo
from agents.tracker import ProgressInfo, create_tracker
from utils.automation.cfg_log_conversion import get_work_dir
from utils.io.config import load_config


_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class BaseJob:
    """Object-oriented representation of a single training job."""

    def __init__(
        self,
        config: str,
        work_dir: str,
        progress: ProgressInfo,
        status: _JobStatus,
        process_info: Optional[ProcessInfo] = None,
    ) -> None:
        self.config = config
        self.work_dir = work_dir
        self.progress = progress
        self.status = status
        self.process_info = process_info

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': self.config,
            'work_dir': self.work_dir,
            'progress': self._serialize(self.progress),
            'status': self.status,
            'process_info': self._serialize(self.process_info),
        }

    @classmethod
    def build(
        cls,
        config: str,
        epochs: int,
        config_to_process_info: Dict[str, ProcessInfo],
        sleep_time: int = 86400,
        outdated_days: int = 30,
        force_progress_recompute: bool = False,
    ) -> 'BaseJob':
        """Construct a BaseJob instance for the provided config."""
        work_dir = get_work_dir(config)
        config_dict = load_config(config)

        tracker = create_tracker(work_dir, config_dict)
        progress = tracker.get_progress(force_progress_recompute=force_progress_recompute)

        log_last_update = cls.get_log_last_update(work_dir, tracker.get_log_pattern())
        epoch_last_update = cls.get_epoch_last_update(work_dir, tracker.get_expected_files())

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
        elif config in config_to_process_info:
            status = 'stuck'
        else:
            status = 'failed'

        process_info = config_to_process_info.get(config)

        return cls(
            config=config,
            work_dir=work_dir,
            progress=progress,
            status=status,
            process_info=process_info,
        )

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

    @staticmethod
    def _serialize(value):
        if value is None:
            return None
        if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            return value.to_dict()
        return value


__all__ = ['BaseJob']
