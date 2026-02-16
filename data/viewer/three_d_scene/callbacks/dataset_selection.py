from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash
from dash.dependencies import Input, Output, State

from data.viewer.three_d_scene.callbacks.common import (
    build_static_grid,
)
from data.viewer.three_d_scene.layout import CAMERA_SELECTOR_ROOT_ID, _make_grid_style

if TYPE_CHECKING:
    from data.viewer.three_d_scene.three_d_scene_viewer import ThreeDSceneViewer


def register_dataset_selection_callbacks(
    app: dash.Dash, viewer: "ThreeDSceneViewer"
) -> None:
    """Handle dataset dropdown selection (options + main view update)."""
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert viewer is not None, "viewer must not be None"

    @app.callback(
        Output("scene-dropdown", "options"),
        Output("image-grid", "children", allow_duplicate=True),
        Output("image-grid", "style", allow_duplicate=True),
        Output(CAMERA_SELECTOR_ROOT_ID, "children", allow_duplicate=True),
        Output("camera-info-display", "children", allow_duplicate=True),
        Output("scene-dropdown", "value", allow_duplicate=True),
        Output("translation-step-display", "children", allow_duplicate=True),
        Output("rotation-step-display", "children", allow_duplicate=True),
        Input("dataset-dropdown", "value"),
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
    def on_dataset_change(
        dataset_value: Optional[str],
        show_cameras_state: bool,
        _store_state_values: Any,
    ) -> Tuple[
        List[Dict[str, str]],
        List[Any],
        Dict[str, Any],
        Any,
        str,
        Optional[str],
        str,
        str,
    ]:
        # Input validations
        assert dataset_value is None or isinstance(
            dataset_value, str
        ), "dataset_value must be str or None"
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"

        viewer.set_dataset(dataset_value)
        if dataset_value is None:
            scene_options: List[Dict[str, str]] = []
        else:
            scene_names = viewer.scene_order[dataset_value]
            scene_options = [{"label": scene, "value": scene} for scene in scene_names]

        grid_children, grid_style = build_static_grid(viewer)
        return (
            scene_options,
            grid_children,
            grid_style,
            None,
            "",
            None,
            "Step: --",
            "Step: --",
        )
