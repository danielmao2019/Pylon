"""Callback for updating monitor resource table."""

from typing import Dict

from dash import Dash, Input, Output

from agents.monitor.callbacks.validation import validate_trigger
from agents.monitor.dashboard import build_style, build_table_rows, format_last_update
from agents.monitor.system_monitor import SystemMonitor


def register_update_table_callback(
    app: Dash, monitors: Dict[str, SystemMonitor]
) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(monitors, dict), f"monitors must be dict, got {type(monitors)}"

    @app.callback(
        Output("resource-table", "data"),
        Output("resource-table", "style_data_conditional"),
        Output("last-update", "children"),
        Input("refresh-interval", "n_intervals"),
    )
    def _update_table(intervals: int):
        # Input validations
        assert isinstance(
            intervals, int
        ), f"intervals must be int, got {type(intervals)}"

        validate_trigger(expected_id="refresh-interval")
        rows = build_table_rows(monitors=monitors)
        style = build_style(rows)
        return rows, style, format_last_update()
