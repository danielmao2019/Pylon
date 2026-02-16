"""Navigation-related callbacks for the viewer."""

from typing import TYPE_CHECKING, List

import dash
from dash.dependencies import Input, Output

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_navigation_current_index_callbacks(
    app: dash.Dash, viewer: "DatasetViewer"
) -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output("current-index-display", "children")],
        [Input("datapoint-index-slider", "value")],
    )
    def update_current_index(current_idx: int) -> List[str]:
        """Update the current index display."""
        # Input validations
        assert isinstance(
            current_idx, int
        ), f"current_idx must be int, got {type(current_idx)}"

        return [f"Current Index: {current_idx}"]
