"""Runtime parameters supplied to manager-backed jobs."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

from agents.monitor.process_info import ProcessInfo


@dataclass(frozen=True)
class JobRuntimeParams:
    """Container for runtime information shared with managed jobs."""

    epochs: int
    sleep_time: int
    command_processes: Mapping[str, ProcessInfo]
    outdated_days: int | None = None
    outdated_date: datetime | None = None
    force_progress_recompute: bool = False

    def process_for(self, command: str) -> Optional[ProcessInfo]:
        """Return an attached process for the given command if one exists."""
        return self.command_processes.get(command)
