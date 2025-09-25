"""Runtime parameters supplied to manager-backed jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from agents.monitor.process_info import ProcessInfo


@dataclass(frozen=True)
class JobRuntimeParams:
    """Container for runtime information shared with managed jobs."""

    epochs: int
    sleep_time: int
    outdated_days: int
    config_processes: Mapping[str, ProcessInfo]
    force_progress_recompute: bool = False

    def process_for(self, config_filepath: str) -> Optional[ProcessInfo]:
        """Return an attached process for the given config if one exists."""
        return self.config_processes.get(config_filepath)

