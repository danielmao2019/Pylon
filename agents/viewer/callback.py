from typing import Dict, List
import datetime
from dash import Input, Output
from agents.viewer.layout import generate_table_data, generate_table_style
from agents.viewer.backend import get_progress
from utils.automation.run_status import get_all_run_status
from utils.monitor.gpu_monitor import GPUMonitor


def register_callbacks(
    app,
    config_files: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    servers: List[str],
    gpu_monitor: GPUMonitor,
    user_names: Dict[str, str],
) -> None:
    """Register callbacks for the dashboard.

    Args:
        app: Dash application instance
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
        sleep_time: Time to wait for the status to update
        outdated_days: Number of days to consider a run outdated
        servers: List of servers
        gpu_monitor: GPUMonitor object
        user_names: Dict of user names
    """
    @app.callback(
        [
            Output('last-update', 'children'),
            Output('progress', 'children'),
            Output('status-table', 'data'),
            Output('status-table', 'style_data_conditional'),
        ],
        Input('interval-component', 'n_intervals')
    )
    def update_table(n_intervals):
        last_update = f"Last Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        progress = f"Progress: {get_progress(config_files, expected_files, epochs, sleep_time, outdated_days, servers)}%"
        table_data = generate_table_data(gpu_monitor, user_names)
        table_style = generate_table_style(table_data)
        return last_update, progress, table_data, table_style
