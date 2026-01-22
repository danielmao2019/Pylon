"""Register callbacks for the Agents monitor Dash app."""

from typing import Dict

from dash import Dash

from agents.monitor.callbacks.update_table import register_update_table_callback
from agents.monitor.system_monitor import SystemMonitor


def register_callbacks(app: Dash, monitors: Dict[str, SystemMonitor]) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(monitors, dict), f"monitors must be dict, got {type(monitors)}"

    register_update_table_callback(app=app, monitors=monitors)
