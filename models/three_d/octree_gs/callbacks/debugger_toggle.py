from typing import Any, List, Optional, Tuple

import dash
import torch
from dash import MATCH
from dash.dependencies import Input, Output, State

from data.structures.three_d.camera.camera import Camera
from models.three_d.octree_gs.callbacks.common import decode_trigger_id


def register_octree_debugger_toggle(app: dash.Dash, viewer: Any) -> None:
    @app.callback(
        Output(
            {
                "type": "model-body",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
            },
            "children",
            allow_duplicate=True,
        ),
        Output(
            {
                "type": "model-store",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
                "field": "octree_debugger_enabled",
            },
            "data",
            allow_duplicate=True,
        ),
        Input(
            {
                "type": "octree-debugger-toggle",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
            },
            "n_clicks",
        ),
        State("camera-overlay-toggle-store", "data"),
        State(
            {
                "type": "model-store",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
                "field": "octree_debugger_enabled",
            },
            "data",
        ),
        State(
            {
                "type": "model-store",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
                "field": "octree_selected_levels_rgb",
            },
            "data",
        ),
        State(
            {
                "type": "model-store",
                "dataset": MATCH,
                "scene": MATCH,
                "method": MATCH,
                "field": "octree_selected_levels_density",
            },
            "data",
        ),
        prevent_initial_call=True,
    )
    def _on_octree_debugger_toggle(
        n_clicks: int,
        show_cameras: bool,
        debugger_enabled: bool,
        rgb_levels_store: Optional[List[Any]],
        density_levels_store: Optional[List[Any]],
    ) -> Tuple[List[Any], bool]:
        assert isinstance(n_clicks, int), f"{type(n_clicks)=}"
        assert isinstance(
            show_cameras, bool
        ), "overlay toggle store data must be a boolean"
        assert isinstance(debugger_enabled, bool), f"{type(debugger_enabled)=}"
        if n_clicks == 0:
            return dash.no_update, debugger_enabled
        assert rgb_levels_store is None or isinstance(
            rgb_levels_store, list
        ), f"{type(rgb_levels_store)=}"
        assert density_levels_store is None or isinstance(
            density_levels_store, list
        ), f"{type(density_levels_store)=}"

        triggered_id = decode_trigger_id()
        dataset_name = triggered_id["dataset"]
        scene_name = triggered_id["scene"]
        method_name = triggered_id["method"]

        assert hasattr(viewer, 'dataset_cache')
        assert dataset_name in viewer.dataset_cache
        assert scene_name in viewer.dataset_cache[dataset_name]
        assert method_name in viewer.method_index[dataset_name][scene_name]
        dataset_instance = viewer.dataset_cache[dataset_name][scene_name]
        method_idx = viewer.method_index[dataset_name][scene_name][method_name]
        datapoint = dataset_instance[method_idx]
        annotation = dataset_instance.annotations[method_idx]
        render_resolution = viewer.get_render_resolution(
            annotation["camera_resolution"]
        )
        display_cameras = (
            viewer.get_scene_camera_overlays(annotation=annotation)
            if show_cameras
            else None
        )

        new_value = not debugger_enabled
        scene_model = datapoint["inputs"]["model"]
        camera_name = None
        if viewer._current_camera_selection is not None:
            assert ":" in viewer._current_camera_selection, (
                f"Camera selection '{viewer._current_camera_selection}' is missing ':' "
                "separator"
            )
            camera_name = viewer._current_camera_selection.split(":", 1)[1]

        render_camera = viewer.get_camera()
        if display_cameras is not None:
            assert isinstance(display_cameras, list), f"{type(display_cameras)=}"
            assert all(isinstance(cam, Camera) for cam in display_cameras)

        component = scene_model.display_render(
            camera=render_camera,
            resolution=render_resolution,
            camera_name=camera_name,
            display_cameras=display_cameras,
            title=None,
            device=dataset_instance.device,
            dataset_name=dataset_name,
            scene_name=scene_name,
            method=method_name,
            octree_selected_levels_rgb=rgb_levels_store,
            octree_selected_levels_density=density_levels_store,
            octree_debugger_enabled=new_value,
        )
        return [component], new_value
