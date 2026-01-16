import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Sequence

from agents.manager.base_progress_info import BaseProgressInfo
from agents.manager.runtime import JobRuntimeParams
from agents.monitor.process_info import ProcessInfo

_JobStatus = Literal['running', 'finished', 'failed', 'stuck', 'outdated']


class BaseJob(ABC):
    """Generic representation of a job launched via an arbitrary command.

    The base class captures the command invocation and shared state needed by
    concrete implementations. Subclasses are responsible for deriving
    progress/status semantics from project-specific artefacts.
    """

    EXPECTED_FILES: Sequence[str] = ()

    def __init__(
        self,
        command: str,
    ) -> None:
        assert (
            isinstance(command, str) and command.strip()
        ), "Job command must be a non-empty string"
        self.command = command
        self.work_dir = self.derive_work_dir()
        self.process_info: Optional[ProcessInfo] = None
        self.progress: Optional[BaseProgressInfo] = None
        self.status: Optional[str] = None

    @abstractmethod
    def derive_work_dir(self) -> str:
        """Return the canonical work directory for this job instance."""

    def configure(self, runtime: JobRuntimeParams) -> None:
        """Attach runtime parameters and recompute state."""
        self.attach_process(runtime.process_for(self.command))
        progress = self.compute_progress(runtime)
        self.progress = progress
        self.status = self.compute_status(progress, runtime)

    def attach_process(self, process_info: Optional[ProcessInfo]) -> None:
        """Associate runtime process information (if available)."""
        self.process_info = process_info

    # ----------------------------------------------------------------------------------------------------
    # compute progress
    # ----------------------------------------------------------------------------------------------------

    @abstractmethod
    def compute_progress(self, runtime: JobRuntimeParams) -> BaseProgressInfo:
        """Return up-to-date progress information for the job."""

    # ----------------------------------------------------------------------------------------------------
    # compute status
    # ----------------------------------------------------------------------------------------------------

    def compute_status(
        self,
        progress: BaseProgressInfo,
        runtime: JobRuntimeParams,
    ) -> str:
        """Determine high-level status (e.g. running/finished/failed)."""
        if self.is_active(runtime):
            return "running"
        if self.is_complete(progress, runtime):
            return "outdated" if self.is_outdated(runtime) else "finished"
        if self.is_stuck(runtime):
            return "stuck"
        return "failed"

    @abstractmethod
    def is_active(self, runtime: JobRuntimeParams) -> bool:
        """Return True while the job is actively progressing."""
        raise NotImplementedError

    def is_outdated(self, runtime: JobRuntimeParams) -> bool:
        if runtime.outdated_days is None and runtime.outdated_date is None:
            return False

        artifact_last_update = self.get_artifact_last_update()
        if artifact_last_update is None:
            return False

        if runtime.outdated_date is not None:
            return artifact_last_update < runtime.outdated_date.timestamp()

        stale_after = runtime.outdated_days * 24 * 60 * 60
        return (time.time() - artifact_last_update) > stale_after

    def is_complete(
        self, progress: BaseProgressInfo, runtime: JobRuntimeParams
    ) -> bool:
        """Return True when the job should count as complete."""
        assert 0.0 <= progress.progress_percentage <= 100.0, (
            f"Progress percentage must be within [0, 100], got "
            f"{progress.progress_percentage}"
        )
        return progress.progress_percentage == 100.0

    @abstractmethod
    def is_stuck(self, runtime: JobRuntimeParams) -> bool:
        """Return True when the job appears hung."""
        raise NotImplementedError

    def get_artifact_last_update(self) -> Optional[float]:
        """Return the most recent modification timestamp across expected files."""
        timestamps = []
        for relative_path in self.EXPECTED_FILES:
            if not relative_path:
                continue
            artifact_path = os.path.join(self.work_dir, relative_path)
            if os.path.isfile(artifact_path):
                timestamps.append(os.path.getmtime(artifact_path))
        return max(timestamps, default=None)

    # ----------------------------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Light-weight serialisable view of the job state."""
        return {
            "command": self.command,
            "work_dir": self.work_dir,
            "progress": self.progress.to_dict() if self.progress else None,
            "status": self.status,
            "process_info": self.process_info.to_dict() if self.process_info else None,
        }

    @abstractmethod
    def tmux_session_name(self) -> str:
        """Return a short identifier to incorporate into tmux session names."""
        raise NotImplementedError
