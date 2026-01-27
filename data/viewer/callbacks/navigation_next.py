"""Navigation-related callbacks for the viewer."""

from typing import TYPE_CHECKING, List, Optional

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.callbacks.validation import validate_trigger

if TYPE_CHECKING:
    from data.viewer.viewer import DatasetViewer


def register_navigation_next_callbacks(app: dash.Dash, viewer: "DatasetViewer") -> None:
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"

    @app.callback(
        [Output("datapoint-index-slider", "value", allow_duplicate=True)],
        [Input("next-btn", "n_clicks")],
        [
            State("datapoint-index-slider", "value"),
            State("datapoint-index-slider", "max"),
        ],
    )
    def update_index_next(
        next_clicks: Optional[int],
        current_value: int,
        max_value: int,
    ) -> List[int]:
        # Input validations
        assert next_clicks is None or isinstance(
            next_clicks, int
        ), f"next_clicks must be int or None, got {type(next_clicks)}"
        assert isinstance(
            current_value, int
        ), f"current_value must be int, got {type(current_value)}"
        assert isinstance(
            max_value, int
        ), f"max_value must be int, got {type(max_value)}"

        if next_clicks is None:
            raise PreventUpdate

        validate_trigger(expected_id="next-btn")
        new_value = min(max_value, current_value + 1)
        return [new_value]
