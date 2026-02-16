"""Navigation-related callbacks for the viewer."""

from typing import TYPE_CHECKING, List, Optional

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.dataset.callbacks.validation import validate_trigger

if TYPE_CHECKING:
    from data.viewer.dataset.viewer import DatasetViewer


def register_navigation_prev_callbacks(app: dash.Dash, viewer: "DatasetViewer") -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output("datapoint-index-slider", "value", allow_duplicate=True)],
        [Input("prev-btn", "n_clicks")],
        [
            State("datapoint-index-slider", "value"),
            State("datapoint-index-slider", "min"),
        ],
    )
    def update_index_prev(
        prev_clicks: Optional[int],
        current_value: int,
        min_value: int,
    ) -> List[int]:
        # Input validations
        assert prev_clicks is None or isinstance(
            prev_clicks, int
        ), f"prev_clicks must be int or None, got {type(prev_clicks)}"
        assert isinstance(
            current_value, int
        ), f"current_value must be int, got {type(current_value)}"
        assert isinstance(
            min_value, int
        ), f"min_value must be int, got {type(min_value)}"

        if prev_clicks is None:
            raise PreventUpdate

        validate_trigger(expected_id="prev-btn")
        new_value = max(min_value, current_value - 1)
        return [new_value]
