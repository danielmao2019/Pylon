from typing import Dict, List, Tuple

import dash

from agents.base_agent import BaseAgent
from agents.viewer.backend import get_progress
from agents.viewer.callbacks import register_callbacks
from agents.viewer.dashboard import (
    format_last_update,
    generate_table_data,
    generate_table_style,
)
from agents.viewer.layout import build_layout


class AgentsViewerApp(BaseAgent):

    def __init__(
        self,
        commands: List[str],
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
            commands=commands,
            expected_files=expected_files,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days,
            gpu_pool=gpu_pool,
            user_names=user_names,
            timeout=timeout,
            force_progress_recompute=force_progress_recompute,
        )
        self.app = dash.Dash(__name__)
        last_update_text = format_last_update()
        progress_value = get_progress(
            commands=commands,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days,
            system_monitors=self.system_monitors,
            force_progress_recompute=force_progress_recompute,
        )
        progress_text = f"Progress: {progress_value}%"
        table_data = generate_table_data(
            system_monitors=self.system_monitors, user_names=user_names
        )
        table_style = generate_table_style(table_data)
        build_layout(
            app=self.app,
            last_update_text=last_update_text,
            progress_text=progress_text,
            table_data=table_data,
            table_style=table_style,
        )
        register_callbacks(
            app=self.app,
            commands=commands,
            expected_files=expected_files,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days,
            system_monitors=self.system_monitors,
            user_names=user_names,
            force_progress_recompute=force_progress_recompute,
        )

    def run(self, port: int = 8050) -> None:
        # Input validations
        assert isinstance(port, int), f"port must be int, got {type(port)}"
        assert 1024 <= port <= 65535, f"port must be between 1024 and 65535, got {port}"

        self.app.run(host="0.0.0.0", port=port, debug=False)
