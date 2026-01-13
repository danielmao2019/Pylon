import json
from typing import Any, Dict, List

import torch
from dash import callback_context
from dash.exceptions import PreventUpdate

from data.structures.three_d.camera.camera import Camera
from data.viewer.utils.atomic_displays.image_display import create_image_display
from models.three_d.octree_gs import styles
from models.three_d.octree_gs.render import (
    render_density_from_octree_gs,
    render_rgb_from_octree_gs,
)
from models.three_d.octree_gs.scene_model import OctreeGSSceneModel


def decode_trigger_id() -> Dict[str, Any]:
    trigger_id = callback_context.triggered_id
    if trigger_id is None:
        raise PreventUpdate
    if isinstance(trigger_id, str):
        decoded = json.loads(trigger_id)
        assert isinstance(
            decoded, dict
        ), f"Triggered id must decode to dict, got {type(decoded)}"
        return decoded
    assert isinstance(
        trigger_id, dict
    ), f"Triggered id must be dict, got {type(trigger_id)}"
    return trigger_id


def build_octree_main_figure(
    viewer: Any,
    dataset_name: str,
    scene_name: str,
    method_name: str,
    modality: str,
    selected_levels: List[int],
    show_cameras: bool,
):
    assert hasattr(viewer, 'dataset_cache')
    assert dataset_name in viewer.dataset_cache
    assert scene_name in viewer.dataset_cache[dataset_name]
    dataset_instance = viewer.dataset_cache[dataset_name][scene_name]
    method_idx = viewer.method_index[dataset_name][scene_name][method_name]
    datapoint = dataset_instance[method_idx]
    scene_data = datapoint["inputs"]["model"].model
    annotation = dataset_instance.annotations[method_idx]
    render_resolution = viewer.get_render_resolution(annotation["camera_resolution"])
    render_camera = viewer.get_camera()
    if show_cameras:
        display_cameras = viewer.get_scene_camera_overlays(annotation=annotation)
    else:
        display_cameras = None

    assert isinstance(
        selected_levels, list
    ), f"selected_levels must be list, got {type(selected_levels)=}"
    if modality == "rgb":
        image = render_rgb_from_octree_gs(
            model=scene_data,
            intrinsics=render_camera.intrinsics,
            extrinsics=render_camera.extrinsics,
            convention=render_camera.convention,
            resolution=render_resolution,
            levels=selected_levels,
            return_info=False,
        )
        label = "RGB"
    else:
        image = render_density_from_octree_gs(
            model=scene_data,
            camera=render_camera,
            resolution=render_resolution,
            levels=selected_levels,
            return_info=False,
            density_color=(0.0, 0.0, 1.0),
            uniform_scale=0.02,
        )
        label = "Density"

    render_camera = render_camera.to(device=image.device)
    if display_cameras is not None:
        assert isinstance(display_cameras, list), f"{type(display_cameras)=}"
        assert all(isinstance(cam, Camera) for cam in display_cameras)

    composed = OctreeGSSceneModel._apply_camera_overlays(
        image=image,
        display_cameras=display_cameras,
        render_at_camera=render_camera,
        resolution=render_resolution,
    )

    fig = create_image_display(
        image=composed,
        title=label,
        colorscale="Viridis",
    )
    fig.update_layout(**styles.figure_layout_with_title(label))
    fig.update_coloraxes(**styles.coloraxis_no_scale())
    return fig
