"""Update plots when the smoothing slider changes."""

from typing import Any, List, Tuple

from dash import Dash, Input, Output

from runners.viewers.train_viewer.callbacks.helpers import (
    LOG_DIRS_FILEPATH,
    build_loss_plots,
    build_smoothing_info,
    load_log_dirs,
)


def register_slider_callback(app: Dash) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"

    @app.callback(
        Output("plots-container", "children", allow_duplicate=True),
        Output("smoothing-info", "children", allow_duplicate=True),
        Input("smoothing-slider", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _update_plots_from_slider(smoothing_window: int) -> Tuple[List[Any], str]:
        # Input validations
        assert isinstance(smoothing_window, int), "smoothing_window must be an integer"

        log_dirs = load_log_dirs(filepath=LOG_DIRS_FILEPATH)
        plots = build_loss_plots(log_dirs=log_dirs, smoothing_window=smoothing_window)
        smoothing_info = build_smoothing_info(smoothing_window=smoothing_window)
        return plots, smoothing_info
