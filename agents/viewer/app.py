from typing import Tuple, List, Dict, Optional
import dash
from utils.monitor.gpu_status import GPUStatus
from utils.monitor.gpu_monitor import GPUMonitor
from agents.viewer.layout import create_layout
from agents.viewer.callback import register_callbacks


class AgentsViewerApp:

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
        epochs: int,
        sleep_time: int = 180,
        outdated_days: int = 120,
        gpu_pool: List[Tuple[str, List[int]]] = [],
        user_names: Dict[str, str] = {},
    ) -> None:
        self._init_gpu_monitor(gpu_pool)
        self.servers = [server for server, _ in gpu_pool]
        self.app = dash.Dash(__name__)
        self.app.layout = create_layout(config_files, expected_files, epochs, sleep_time, outdated_days, self.servers, self.gpu_monitor, user_names)
        register_callbacks(self.app, config_files, expected_files, epochs, sleep_time, outdated_days, self.servers, self.gpu_monitor, user_names)

    def _init_gpu_monitor(self, gpu_pool: List[Tuple[str, List[int]]]) -> None:
        gpus = [
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
            for server, indices in gpu_pool
            for idx in indices
        ]
        self.gpu_monitor = GPUMonitor(gpus)
        self.gpu_monitor.start()

    def run(self, port: Optional[int] = 8050) -> None:
        self.app.run(port=port)
