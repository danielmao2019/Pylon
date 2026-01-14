from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.ivision.callbacks.common import (
    compute_outputs,
    model_state_from_context,
)
from data.viewer.ivision.layout import CAMERA_SELECTOR_ROOT_ID
from data.viewer.ivision.layout.components import (
    CAMERA_OVERLAY_TOGGLE_BUTTON_ID,
    CAMERA_OVERLAY_TOGGLE_STORE_ID,
)

if TYPE_CHECKING:
    from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer


def register_camera_overlay_toggle_callbacks(
    app: dash.Dash, viewer: "iVISION_4D_Scene_Viewer"
) -> None:
    """Toggle rendering of camera overlays."""

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
        Output(CAMERA_OVERLAY_TOGGLE_STORE_ID, "data"),
        Output(CAMERA_OVERLAY_TOGGLE_BUTTON_ID, "children"),
        Input(CAMERA_OVERLAY_TOGGLE_BUTTON_ID, "n_clicks"),
        State(CAMERA_OVERLAY_TOGGLE_STORE_ID, "data"),
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
    def on_toggle(
        _clicks: int,
        show_cameras_state: bool,
        _store_state_values: Any,
    ) -> Tuple[List[Any], Any, str, bool, str]:
        assert isinstance(_clicks, int), "_clicks must be an int"
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"

        if viewer.current_dataset is None or viewer.current_scene is None:
            raise PreventUpdate

        new_value = not show_cameras_state

        model_state = model_state_from_context()
        body_updates, camera_selector_children, camera_info = compute_outputs(
            viewer=viewer,
            model_state_entries=model_state,
            show_cameras=new_value,
        )
        button_label = "Hide Cameras" if new_value else "Show Cameras"
        return (
            body_updates,
            camera_selector_children,
            camera_info,
            new_value,
            button_label,
        )
