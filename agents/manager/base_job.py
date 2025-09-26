from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from agents.manager.progress_info import ProgressInfo
from agents.monitor.process_info import ProcessInfo
from agents.manager.runtime import JobRuntimeParams


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
        assert isinstance(command, str) and command.strip(), (
            "Job command must be a non-empty string"
        )
        self.command = command
        self.work_dir = self.derive_work_dir()
        self.process_info: Optional[ProcessInfo] = None
        self.progress: Optional[ProgressInfo] = None
        self.status: Optional[str] = None
        self._runtime: Optional[JobRuntimeParams] = None

    def configure(self, runtime: JobRuntimeParams) -> None:
        """Attach runtime parameters and recompute state."""
        self._runtime = runtime
        self.attach_process(runtime.process_for(self.config_filepath))
        progress = self.compute_progress()
        self.progress = progress
        self.status = self.compute_status(progress)

    def attach_process(self, process_info: Optional[ProcessInfo]) -> None:
        """Associate runtime process information (if available)."""
        self.process_info = process_info

    @abstractmethod
    def compute_progress(self) -> ProgressInfo:
        """Return up-to-date progress information for the job."""

    def compute_status(self, progress: ProgressInfo) -> str:
        """Determine high-level status (e.g. running/finished/failed)."""
        if self.is_active():
            return "running"
        if self.is_complete(progress):
            return "outdated" if self.is_outdated() else "finished"
        if self.is_stuck():
            return "stuck"
        return "failed"

    @property
    def runtime(self) -> JobRuntimeParams:
        if self._runtime is None:
            raise RuntimeError("Job runtime has not been configured yet")
        return self._runtime

    def describe(self) -> str:
        """Human-friendly descriptor, defaulting to the launch command."""
        return self.command

    @abstractmethod
    def derive_work_dir(self) -> str:
        """Return the canonical work directory for this job instance."""

    def to_dict(self) -> Dict[str, Any]:
        """Light-weight serialisable view of the job state."""
        return {
            "command": self.command,
            "work_dir": self.work_dir,
            "progress": self.progress.to_dict() if self.progress else None,
            "status": self.status,
            "process_info": self.process_info.to_dict() if self.process_info else None,
        }

    # ------------------------------------------------------------------
    # Status helper hooks (overridable by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def is_active(self) -> bool:
        """Return True while the job is actively progressing."""
        raise NotImplementedError

    def is_outdated(self) -> bool:
        runtime = self._runtime
        if runtime is None or runtime.outdated_days <= 0:
            return False

        artifact_last_update = self.get_artifact_last_update()
        if artifact_last_update is None:
            return False

        stale_after = runtime.outdated_days * 24 * 60 * 60
        return (time.time() - artifact_last_update) > stale_after

    @abstractmethod
    def is_complete(self, progress: ProgressInfo) -> bool:
        """Return True when the job should count as complete."""
        raise NotImplementedError

    @abstractmethod
    def is_stuck(self) -> bool:
        """Return True when the job appears hung."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared artifact helpers
    # ------------------------------------------------------------------

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
