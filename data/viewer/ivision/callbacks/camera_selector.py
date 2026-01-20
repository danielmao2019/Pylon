from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.ivision.callbacks.common import (
    compute_outputs,
    model_state_from_context,
)
from data.viewer.ivision.callbacks.helpers import triggered_payload
from data.viewer.ivision.layout import (
    CAMERA_NONE_VALUE,
    CAMERA_SELECTOR_RADIO_TYPE,
    CAMERA_SELECTOR_ROOT_ID,
)

if TYPE_CHECKING:
    from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer


def register_camera_selector_callbacks(
    app: dash.Dash, viewer: "iVISION_4D_Scene_Viewer"
) -> None:
    """Handle camera selector changes."""

    @app.callback(
        Output(
            {
                "type": "model-body",
                "dataset": dash.dependencies.ALL,
                "scene": dash.dependencies.ALL,
                "method": dash.dependencies.ALL,
            },
            "children",
            allow_duplicate=True,
        ),
        Output(CAMERA_SELECTOR_ROOT_ID, "children", allow_duplicate=True),
        Output("camera-info-display", "children", allow_duplicate=True),
        Output("scene-dropdown", "value", allow_duplicate=True),
        Input(
            {
                "type": CAMERA_SELECTOR_RADIO_TYPE,
                "dataset": dash.dependencies.ALL,
                "scene": dash.dependencies.ALL,
                "split": dash.dependencies.ALL,
            },
            "value",
        ),
        State("camera-overlay-toggle-store", "data"),
        State(
            {
                "type": "model-store",
                "dataset": dash.dependencies.ALL,
                "scene": dash.dependencies.ALL,
                "method": dash.dependencies.ALL,
                "field": dash.dependencies.ALL,
            },
            "data",
        ),
        prevent_initial_call=True,
    )
    def on_camera_select(
        _camera_values: Any,
        show_cameras_state: bool,
        _store_state_values: Any,
    ) -> Tuple[List[Any], Any, str, Any]:
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"
        triggered_id, triggered_value = triggered_payload()
        if (
            not isinstance(triggered_id, dict)
            or triggered_id.get("type") != CAMERA_SELECTOR_RADIO_TYPE
        ):
            raise PreventUpdate
        selection_guard = triggered_value == viewer._current_camera_selection or (
            triggered_value == CAMERA_NONE_VALUE
            and viewer._current_camera_selection is None
        )
        if selection_guard:
            raise PreventUpdate
        viewer.set_camera_by_selection(triggered_value)
        model_state = model_state_from_context()
        body_updates, camera_selector_children, camera_info = compute_outputs(
            viewer=viewer,
            model_state_entries=model_state,
            show_cameras=show_cameras_state,
        )
        return (
            body_updates,
            camera_selector_children,
            camera_info,
            dash.no_update,
        )
