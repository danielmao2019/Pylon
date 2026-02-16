from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from data.viewer.three_d_scene.callbacks.common import (
    build_static_grid,
    compute_outputs,
    model_state_from_context,
)
from data.viewer.three_d_scene.layout import CAMERA_SELECTOR_ROOT_ID

if TYPE_CHECKING:
    from data.viewer.three_d_scene.three_d_scene_viewer import ThreeDSceneViewer


def register_scene_selection_callbacks(
    app: dash.Dash, viewer: "ThreeDSceneViewer"
) -> None:
    """Handle scene dropdown selection (main view update)."""
    # Input validations
    assert isinstance(app, dash.Dash), f"app must be Dash, got {type(app)}"
    assert viewer is not None, "viewer must not be None"

    @app.callback(
        Output("image-grid", "children", allow_duplicate=True),
        Output("image-grid", "style", allow_duplicate=True),
        Output(CAMERA_SELECTOR_ROOT_ID, "children", allow_duplicate=True),
        Output("camera-info-display", "children", allow_duplicate=True),
        Output("scene-dropdown", "value", allow_duplicate=True),
        Output("translation-step-display", "children", allow_duplicate=True),
        Output("rotation-step-display", "children", allow_duplicate=True),
        Input("scene-dropdown", "value"),
        State("dataset-dropdown", "value"),
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
    def on_scene_change(
        scene_value: Optional[str],
        dataset_value: Optional[str],
        translation_slider_value: Optional[float],
        rotation_slider_value: Optional[float],
        show_cameras_state: bool,
        _store_state_values: List[Any],
    ) -> Tuple[
        List[Any],
        Dict[str, Any],
        Any,
        str,
        Optional[str],
        str,
        str,
    ]:
        # Input validations
        assert scene_value is None or isinstance(
            scene_value, str
        ), "scene_value must be str or None"
        assert dataset_value is None or isinstance(
            dataset_value, str
        ), "dataset_value must be str or None"
        assert translation_slider_value is None or isinstance(
            translation_slider_value, (float, int)
        ), "translation_slider_value must be float or int"
        assert rotation_slider_value is None or isinstance(
            rotation_slider_value, (float, int)
        ), "rotation_slider_value must be float or int"
        assert isinstance(
            show_cameras_state, bool
        ), "overlay toggle store data must be a boolean"

        if scene_value is None or dataset_value is None:
            raise PreventUpdate
        viewer.set_dataset(dataset_value)
        viewer.set_scene(scene_value)
        model_state = model_state_from_context()
        (
            body_updates,
            camera_selector_children,
            camera_info,
        ) = compute_outputs(
            viewer=viewer,
            model_state_entries=model_state,
            show_cameras=show_cameras_state,
        )
        grid_children, grid_style = build_static_grid(viewer)
        assert len(grid_children) == len(body_updates), (
            f"Static grid children ({len(grid_children)}) must equal body updates "
            f"({len(body_updates)})"
        )
        populated_children: List[Any] = []
        for container, body in zip(grid_children, body_updates):
            container_children = container.children
            assert isinstance(
                container_children, list
            ), "Static container children must be a list"
            updated_children: List[Any] = []
            body_inserted = False
            for child in container_children:
                child_id = getattr(child, "id", None)
                if isinstance(child_id, dict) and child_id.get("type") == "model-body":
                    child.children = [body]
                    updated_children.append(child)
                    body_inserted = True
                else:
                    updated_children.append(child)
            assert body_inserted, "Static container missing model-body placeholder"
            container.children = updated_children
            populated_children.append(container)
        translation_step = viewer.get_translation_step(translation_slider_value)
        rotation_step = viewer.get_rotation_step(rotation_slider_value)
        translation_display = f"Step: {translation_step:.2f}"
        rotation_display = f"Step: {rotation_step:.2f}°"
        return (
            populated_children,
            grid_style,
            camera_selector_children,
            camera_info,
            scene_value,
            translation_display,
            rotation_display,
        )
