"""Register callbacks for the training losses viewer."""

from dash import Dash

from runners.viewers.train_viewer.callbacks.update_plots_from_refresh import (
    register_refresh_callback,
)
from runners.viewers.train_viewer.callbacks.update_plots_from_slider import (
    register_slider_callback,
)


def register_callbacks(app: Dash) -> None:
    # Input validations
    assert isinstance(app, Dash), f"app must be Dash, got {type(app)}"

    register_refresh_callback(app=app)
    register_slider_callback(app=app)
