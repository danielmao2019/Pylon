"""Callback for updating the Agents viewer status table."""

from typing import Any, Dict, List, Tuple

import dash
from dash import Input, Output

from agents.monitor.system_monitor import SystemMonitor
from agents.viewer.backend import get_progress
from agents.viewer.dashboard import (
    format_last_update,
    generate_table_data,
    generate_table_style,
)


def register_update_table_callback(
    app: dash.Dash,
    commands: List[str],
    expected_files: List[str],
    epochs: int,
    sleep_time: int,
    outdated_days: int,
    system_monitors: Dict[str, SystemMonitor],
    user_names: Dict[str, str],
    force_progress_recompute: bool = False,
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert isinstance(commands, list), f"commands must be list, got {type(commands)}"
    assert commands, "commands must be non-empty"
    assert isinstance(
        expected_files, list
    ), f"expected_files must be list, got {type(expected_files)}"
    assert expected_files, "expected_files must be non-empty"
    assert isinstance(epochs, int), f"epochs must be int, got {type(epochs)}"
    assert isinstance(
        sleep_time, int
    ), f"sleep_time must be int, got {type(sleep_time)}"
    assert isinstance(
        outdated_days, int
    ), f"outdated_days must be int, got {type(outdated_days)}"
    assert isinstance(
        system_monitors, dict
    ), f"system_monitors must be dict, got {type(system_monitors)}"
    assert system_monitors, "system_monitors must be non-empty"
    assert all(
        isinstance(monitor, SystemMonitor) for monitor in system_monitors.values()
    ), "system_monitors must contain SystemMonitor values"
    assert isinstance(
        user_names, dict
    ), f"user_names must be dict, got {type(user_names)}"
    assert isinstance(
        force_progress_recompute, bool
    ), f"force_progress_recompute must be bool, got {type(force_progress_recompute)}"

    @app.callback(
        [
            Output("last-update", "children"),
            Output("progress", "children"),
            Output("status-table", "data"),
            Output("status-table", "style_data_conditional"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def update_table(
        n_intervals: int,
    ) -> Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Input validations
        assert isinstance(
            n_intervals, int
        ), f"n_intervals must be int, got {type(n_intervals)}"

        last_update = format_last_update()
        progress_value = get_progress(
            commands=commands,
            epochs=epochs,
            sleep_time=sleep_time,
            outdated_days=outdated_days,
            system_monitors=system_monitors,
            force_progress_recompute=force_progress_recompute,
        )
        progress = f"Progress: {progress_value}%"
        table_data = generate_table_data(
            system_monitors=system_monitors, user_names=user_names
        )
        table_style = generate_table_style(table_data)
        return last_update, progress, table_data, table_style
