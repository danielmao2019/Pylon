from abc import ABC
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from agents.monitor.cpu_status import CPUStatus
from agents.monitor.gpu_status import GPUStatus
from agents.monitor.system_monitor import SystemMonitor


class BaseAgent(ABC):

    def __init__(
        self,
        commands: List[str],
        expected_files: Optional[List[str]] = None,
        epochs: int = 100,
        sleep_time: int = 180,
        outdated_days: int | None = None,
        outdated_date: datetime | None = None,
        gpu_pool: List[Tuple[str, List[int]]] = [],
        user_names: Dict[str, str] = {},
        timeout: int = 5,
        force_progress_recompute: bool = False,
    ) -> None:
        assert isinstance(
            commands, list
        ), f"commands must be list, got {type(commands)}"
        assert outdated_days is None or (
            isinstance(outdated_days, int) and outdated_days > 0
        ), f"outdated_days must be positive int or None, got {outdated_days}"
        assert outdated_date is None or isinstance(
            outdated_date, datetime
        ), f"outdated_date must be datetime or None, got {type(outdated_date)}"
        assert not (
            outdated_days is not None and outdated_date is not None
        ), "outdated_days and outdated_date cannot both be set"
        self.commands = [command.strip() for command in commands]
        self.expected_files = expected_files or []
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self.outdated_date = outdated_date
        self.force_progress_recompute = force_progress_recompute
        self._init_system_monitors(gpu_pool, timeout)
        self.user_names = user_names

    def _init_system_monitors(
        self, gpu_pool: List[Tuple[str, List[int]]], timeout: int
    ) -> None:
        self.system_monitors: Dict[str, SystemMonitor] = {}
        for server, indices in gpu_pool:
            monitor = SystemMonitor(server=server, gpu_indices=indices, timeout=timeout)
            monitor.start()
            self.system_monitors[server] = monitor
        self.servers = list(self.system_monitors.keys())

    @property
    def connected_gpus(self) -> List[GPUStatus]:
        return [
            gpu
            for monitor in self.system_monitors.values()
            for gpu in monitor.connected_gpus
        ]

    @property
    def connected_cpus(self) -> List[CPUStatus]:
        return [
            monitor.cpu
            for monitor in self.system_monitors.values()
            if monitor.cpu.connected
        ]

    @property
    def disconnected_gpus(self) -> Dict[str, List[int]]:
        return {
            server: monitor.disconnected_gpus
            for server, monitor in self.system_monitors.items()
            if monitor.disconnected_gpus
        }

    @property
    def disconnected_cpus(self) -> List[str]:
        return [
            server
            for server, monitor in self.system_monitors.items()
            if monitor.disconnected_cpu
        ]
