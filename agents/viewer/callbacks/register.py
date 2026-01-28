"""Register callbacks for the Agents viewer Dash app."""

from typing import Dict, List

import dash

from agents.monitor.system_monitor import SystemMonitor
from agents.viewer.callbacks.update_table import register_update_table_callback


def register_callbacks(
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
    assert isinstance(
        expected_files, list
    ), f"expected_files must be list, got {type(expected_files)}"
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

    register_update_table_callback(
        app=app,
        commands=commands,
        expected_files=expected_files,
        epochs=epochs,
        sleep_time=sleep_time,
        outdated_days=outdated_days,
        system_monitors=system_monitors,
        user_names=user_names,
        force_progress_recompute=force_progress_recompute,
    )
