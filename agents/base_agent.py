from typing import Tuple, List, Dict
from abc import ABC
from utils.monitor.gpu_status import GPUStatus
from utils.monitor.gpu_monitor import GPUMonitor


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
        self._init_gpu_monitor(gpu_pool, timeout)
        self.user_names = user_names

    def _init_gpu_monitor(self, gpu_pool: List[Tuple[str, List[int]]], timeout: int) -> None:
        # Create GPU status objects organized by server
        gpus_by_server = {}
        for server, indices in gpu_pool:
            gpus_by_server[server] = [
                GPUStatus(
                    server=server,
                    index=idx,
                    max_memory=0,  # Will be populated by monitor
                    processes=[],
                    window_size=10,
                    memory_window=[],
                    util_window=[],
                    memory_stats={'min': None, 'max': None, 'avg': None},
                    util_stats={'min': None, 'max': None, 'avg': None}
                )
                for idx in indices
            ]

        self.gpu_monitor = GPUMonitor(gpus_by_server, timeout=timeout)
        self.gpu_monitor.start()
        self.servers = [server for server, _ in gpu_pool]
