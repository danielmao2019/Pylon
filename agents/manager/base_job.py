from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from agents.manager.progress_info import ProgressInfo
from agents.monitor.process_info import ProcessInfo


class BaseJob(ABC):
    """Generic representation of a job launched via an arbitrary command.

    The base class captures the command invocation, optional working directory,
    ambient environment variables, and metadata needed by concrete
    implementations. Subclasses are responsible for deriving progress/status
    semantics from project-specific artefacts.
    """

    def __init__(
        self,
        command: str,
        *,
        env: Optional[Mapping[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert isinstance(command, str) and command.strip(), (
            "Job command must be a non-empty string"
        )
        self.command = command
        self.work_dir = self.derive_work_dir()
        self.env = dict(env or {})
        self.metadata = dict(metadata or {})
        self.process_info: Optional[ProcessInfo] = None
        self.progress: Optional[ProgressInfo] = None
        self.status: Optional[str] = None

    def attach_process(self, process_info: Optional[ProcessInfo]) -> None:
        """Associate runtime process information (if available)."""
        self.process_info = process_info

    def refresh(self) -> None:
        """Recompute progress and status using subclass-specific logic."""
        progress = self.compute_progress()
        self.progress = progress
        self.status = self.compute_status(progress=progress)

    @abstractmethod
    def compute_progress(self) -> ProgressInfo:
        """Return up-to-date progress information for the job."""

    @abstractmethod
    def compute_status(self, progress: ProgressInfo) -> str:
        """Determine high-level status (e.g. running/finished/failed)."""

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
            "env": self.env,
            "metadata": self.metadata,
            "progress": self.progress.to_dict() if self.progress else None,
            "status": self.status,
            "process_info": self.process_info.to_dict() if self.process_info else None,
        }
