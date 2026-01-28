from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.ivision.callbacks.common import (
    compute_outputs,
    model_state_from_context,
)
from data.viewer.ivision.layout import CAMERA_SELECTOR_ROOT_ID

if TYPE_CHECKING:
    from data.viewer.ivision.ivision_4d_scene_viewer import iVISION_4D_Scene_Viewer


def register_keyboard_callbacks(
    app: dash.Dash, viewer: "iVISION_4D_Scene_Viewer"
) -> None:
    """Handle keyboard navigation and movement."""
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert viewer is not None, "viewer must not be None"

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
        Input("keyboard", "n_keydowns"),
        State("keyboard", "keydown"),
        State("translation-step-slider", "value"),
        State("rotation-step-slider", "value"),
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
    def on_keyboard(
        _n_keydowns: Optional[int],
        keydown: Optional[Dict[str, Any]],
        translation_slider_value: Optional[float],
        rotation_slider_value: Optional[float],
        show_cameras_state: bool,
        _store_state_values: Any,
    ) -> Tuple[List[Any], Any, str]:
        # Input validations
        assert _n_keydowns is None or isinstance(
            _n_keydowns, int
        ), "_n_keydowns must be int or None"
        assert keydown is None or isinstance(
            keydown, dict
        ), "keydown must be dict or None"
        assert translation_slider_value is None or isinstance(
            translation_slider_value, (float, int)
        ), "translation_slider_value must be float or int"
        assert rotation_slider_value is None or isinstance(
            rotation_slider_value, (float, int)
        ), "rotation_slider_value must be float or int"
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"

        if keydown is None:
            raise PreventUpdate
        key_value = keydown.get("key")
        assert isinstance(key_value, str), "keydown key must be a string"
        assert isinstance(_n_keydowns, int), "_n_keydowns must be an int"

        if viewer.current_dataset is None or viewer.current_scene is None:
            raise PreventUpdate

        translation_step = viewer.get_translation_step(translation_slider_value)
        rotation_step = viewer.get_rotation_step(rotation_slider_value)
        viewer.set_pose_from_keyboard(
            key=key_value,
            translation_step=translation_step,
            rotation_step=rotation_step,
        )

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
        )
