"""Application factory for the Agents monitor Dash app."""

from typing import Dict

from dash import Dash

from agents.monitor.callbacks import register_callbacks
from agents.monitor.dashboard import build_meta_text, build_table_rows
from agents.monitor.layout import build_layout
from agents.monitor.system_monitor import SystemMonitor


def create_app(monitors: Dict[str, SystemMonitor], interval_ms: int) -> Dash:
    # Input validations
    assert isinstance(monitors, dict), f"monitors must be dict, got {type(monitors)}"
    assert isinstance(
        interval_ms, int
    ), f"interval_ms must be int, got {type(interval_ms)}"

    meta_text = build_meta_text(monitors=monitors, interval_ms=interval_ms)
    initial_rows = build_table_rows(monitors=monitors)
    app = Dash(__name__)
    build_layout(
        app=app,
        meta_text=meta_text,
        initial_rows=initial_rows,
        interval_ms=interval_ms,
    )
    register_callbacks(app=app, monitors=monitors)
    return app
