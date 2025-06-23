from typing import Tuple, List, Dict
from abc import ABC
from utils.monitor.system_monitor import SystemMonitor


class BaseAgent(ABC):

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
        epochs: int = 100,
        sleep_time: int = 180,
        outdated_days: int = 120,
        gpu_pool: List[Tuple[str, List[int]]] = [],
        user_names: Dict[str, str] = {},
        timeout: int = 5,
    ) -> None:
        self.config_files = config_files
        self.expected_files = expected_files
        self.epochs = epochs
        self.sleep_time = sleep_time
        self.outdated_days = outdated_days
        self._init_system_monitor(gpu_pool, timeout)
        self.user_names = user_names

    def _init_system_monitor(self, gpu_pool: List[Tuple[str, List[int]]], timeout: int) -> None:
        # Convert gpu_pool format to gpu_indices_by_server format
        servers = [x[0] for x in gpu_pool]
        assert len(set(servers)) == len(servers), f"{servers=}"
        gpu_indices_by_server = {}
        cpu_servers = []
        for server, indices in gpu_pool:
            gpu_indices_by_server[server] = indices
            cpu_servers.append(server)

        self.system_monitor = SystemMonitor(gpu_indices_by_server, cpu_servers=cpu_servers, timeout=timeout)
        self.system_monitor.start()
        self.servers = [server for server, _ in gpu_pool]
