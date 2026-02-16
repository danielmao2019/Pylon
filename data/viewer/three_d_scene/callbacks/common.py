from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash
from dash import callback_context

from data.viewer.three_d_scene.callbacks.helpers import parse_model_state
from data.viewer.three_d_scene.layout import (
    CAMERA_SELECTOR_RADIO_TYPE,
    _build_camera_selector_layout,
    _make_grid_style,
    format_camera_info_text,
)

if TYPE_CHECKING:
    from data.viewer.three_d_scene.three_d_scene_viewer import ThreeDSceneViewer


def compute_outputs(
    viewer: "ThreeDSceneViewer",
    model_state_entries: List[Any],
    show_cameras: bool,
) -> Tuple[List[Any], Any, str]:
    # Input validations
    assert viewer is not None, "viewer must not be None"
    assert isinstance(model_state_entries, list), f"{type(model_state_entries)=}"
    assert isinstance(show_cameras, bool), f"{type(show_cameras)=}"

    model_state = parse_model_state(model_state_entries)
    bodies = viewer.get_render_current_scene(
        model_state=model_state, show_cameras=show_cameras
    )

    body_updates: List[Any] = []

    if viewer.current_dataset is not None and viewer.current_scene is not None:
        method_names = viewer.method_order[viewer.current_dataset][viewer.current_scene]
        for method_name in method_names:
            body = bodies.get(method_name)
            assert body is not None, f"Missing body for method {method_name}"
            body_updates.append(body)

    camera_selector_children = None
    camera_info = ""
    selector_options = viewer.get_camera_selector_options()
    if selector_options is not None:
        camera_selector_children = _build_camera_selector_layout(
            splits=selector_options["splits"],
            choice=selector_options["selection"],
            dataset=selector_options["dataset"],
            scene=selector_options["scene"],
        )
        camera_info = format_camera_info_text(viewer.get_camera_info())

    return body_updates, camera_selector_children, camera_info


def build_static_grid(
    viewer: "ThreeDSceneViewer",
) -> Tuple[List[Any], Dict[str, Any]]:
    # Input validations
    assert viewer is not None, "viewer must not be None"

    if viewer.current_dataset is None or viewer.current_scene is None:
        return [], _make_grid_style(method_count=0)

    assert viewer.current_dataset in viewer._static_model_layouts
    assert viewer.current_scene in viewer._static_model_layouts[viewer.current_dataset]
    static_scene = viewer._static_model_layouts[viewer.current_dataset][viewer.current_scene]
    method_names = viewer.method_order[viewer.current_dataset][viewer.current_scene]
    grid_children: List[Any] = []
    for method_name in method_names:
        assert method_name in static_scene, (
            f"Missing static container for {viewer.current_dataset}/"
            f"{viewer.current_scene}/{method_name}"
        )
        grid_children.append(static_scene[method_name])
    grid_style = _make_grid_style(method_count=len(method_names))
    return grid_children, grid_style


def model_state_from_context() -> List[Any]:
    return callback_context.states_list[-1:]


def camera_radio_input() -> Dict[str, Any]:
    return {
        "type": CAMERA_SELECTOR_RADIO_TYPE,
        "dataset": dash.dependencies.ALL,
        "scene": dash.dependencies.ALL,
        "split": dash.dependencies.ALL,
    }
