from typing import Tuple, List, Dict, Optional
import dash
from agents.base_agent import BaseAgent
from agents.viewer.callback import register_callbacks
from agents.viewer.layout import create_layout


class AgentsViewerApp(BaseAgent):

    def __init__(
        self,
        config_files: List[str],
        expected_files: List[str],
        epochs: int,
        sleep_time: int = 180,
        outdated_days: int = 120,
        gpu_pool: List[Tuple[str, List[int]]] = [],
        user_names: Dict[str, str] = {},
        timeout: int = 5,
        force_progress_recompute: bool = False,
    ) -> None:
        super(AgentsViewerApp, self).__init__(
            config_files, expected_files, epochs, sleep_time, outdated_days, gpu_pool, user_names, timeout, force_progress_recompute,
        )
        self.app = dash.Dash(__name__)
        self.app.layout = create_layout(config_files, expected_files, epochs, sleep_time, outdated_days, self.system_monitor, user_names, force_progress_recompute)
        register_callbacks(self.app, config_files, expected_files, epochs, sleep_time, outdated_days, self.system_monitor, user_names, force_progress_recompute)

    def run(self, port: Optional[int] = 8050) -> None:
        self.app.run(port=port)
