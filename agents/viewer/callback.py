from typing import Dict, List, Any
import datetime
from dash import Input, Output
from agents.viewer.layout import generate_table_data, generate_table_style
from agents.viewer.backend import get_progress
from utils.monitor.gpu_monitor import GPUMonitor


def register_callbacks(app, config_files: List[str], expected_files: List[str], epochs: int, gpu_monitor: GPUMonitor):
    """Register callbacks for the dashboard.
    
    Args:
        app: Dash application instance
        config_files: List of config file paths
        expected_files: List of expected file patterns
        epochs: Total number of epochs
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
        progress = f"Progress: {get_progress(config_files, expected_files, epochs)}%"
        table_data = generate_table_data(gpu_monitor, user_names)
        table_style = generate_table_style(table_data)
        return last_update, progress, table_data, table_style
