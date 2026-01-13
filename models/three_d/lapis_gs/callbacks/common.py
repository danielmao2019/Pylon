import json
from typing import Any, Dict, List, Optional

from dash import callback_context
from dash.exceptions import PreventUpdate

from data.structures.three_d.camera.camera import Camera
from data.viewer.utils.atomic_displays.image_display import create_image_display
from models.three_d.lapis_gs import styles
from models.three_d.lapis_gs.render import (
    render_density_from_lapis_gs,
    render_rgb_from_lapis_gs,
)
from models.three_d.lapis_gs.scene_model import LapisGSSceneModel


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


def total_layers(
    viewer: Any, dataset_name: str, scene_name: str, method_name: str
) -> int:
    assert hasattr(viewer, 'dataset_cache')
    assert dataset_name in viewer.dataset_cache
    assert scene_name in viewer.dataset_cache[dataset_name]
    dataset_instance = viewer.dataset_cache[dataset_name][scene_name]
    method_idx = viewer.method_index[dataset_name][scene_name][method_name]
    scene = dataset_instance[method_idx]["inputs"]["model"].model
    split_points = scene.split_points
    total = len(split_points) - 1
    assert total > 0, "LapisGS model must contain at least one layer"
    return total


def sanitize_layers(
    selected: Optional[List[int]],
    stored: Optional[Any],
    total: int,
) -> List[int]:
    raw_value: Optional[Any] = selected if selected is not None else stored
    if raw_value is None:
        return list(range(total))
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    sanitized = sorted({int(val) for val in values if 0 <= int(val) < total})
    return sanitized if sanitized else list(range(total))


def build_lapis_main_figure(
    viewer: Any,
    dataset_name: str,
    scene_name: str,
    method_name: str,
    modality: str,
    selected_layers: List[int],
    show_cameras: bool,
) -> Any:
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
    display_cameras: Optional[List[Camera]] = (
        viewer.get_scene_camera_overlays(annotation=annotation)
        if show_cameras
        else None
    )

    if modality == "rgb":
        image = render_rgb_from_lapis_gs(
            model=scene_data,
            camera=render_camera,
            resolution=render_resolution,
            layers=selected_layers,
            return_info=False,
            device=dataset_instance.device,
        )
        label = "RGB"
    else:
        image = render_density_from_lapis_gs(
            model=scene_data,
            camera=render_camera,
            resolution=render_resolution,
            layers=selected_layers,
            return_info=False,
            density_color=(0.0, 0.0, 1.0),
            uniform_scale=0.02,
            device=dataset_instance.device,
        )
        label = "Density"

    composed = LapisGSSceneModel._apply_camera_overlays(
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
