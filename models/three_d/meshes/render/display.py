from typing import Any, Dict, List, Optional, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.meshes.render.core import (
    render_rgb_from_mesh as render_rgb_from_mesh_func,
)


def render_display(
    scene_model: BaseSceneModel,
    camera: Camera,
    resolution: Optional[Tuple[int, int]],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    device: Optional[torch.device],
) -> Dict[str, Any]:
    """Render the mesh scene for display.

    Args:
        scene_model: Scene model whose mesh is rendered.
        camera: Camera used for the display render.
        resolution: Optional render resolution `(height, width)`; `None` uses the camera's own resolution.
        camera_name: Optional cache key for a rendered snapshot.
        display_cameras: Optional camera frustums to overlay on the rendered image.
        title: Optional display title.
        device: Optional device where the render should run.

    Returns:
        Display payload with image and title fields.
    """

    def _validate_inputs() -> None:
        assert isinstance(scene_model, BaseSceneModel), (
            "Expected `scene_model` to be a BaseSceneModel instance. "
            f"{type(scene_model)=}"
        )
        assert isinstance(camera, Camera), (
            "Expected `camera` to be a Camera instance. " f"{type(camera)=}"
        )
        assert resolution is None or (
            isinstance(resolution, tuple) and len(resolution) == 2
        ), f"Expected `resolution` to be None or a length-2 tuple. {resolution=}"
        if resolution is not None:
            for axis_name, value in zip(("height", "width"), resolution, strict=True):
                assert isinstance(value, (int, torch.Tensor)) and not isinstance(
                    value, bool
                ), (
                    "Expected each render resolution value to be an int or scalar tensor. "
                    f"{axis_name=} {type(value)=} {resolution=}"
                )
                if isinstance(value, torch.Tensor):
                    assert value.numel() == 1, (
                        "Expected tensor render resolution values to contain one value. "
                        f"{axis_name=} {value.shape=}"
                    )
                    assert value.dtype != torch.bool, (
                        "Expected tensor render resolution values not to be bool. "
                        f"{axis_name=} {value.dtype=} {resolution=}"
                    )
                    assert not value.detach().is_floating_point() or torch.equal(
                        value.detach(), torch.round(value.detach())
                    ), (
                        "Expected tensor render resolution values to be integer-valued. "
                        f"{axis_name=} {value=}"
                    )
                    assert bool((value.detach() > 0).cpu().item()), (
                        "Expected tensor render resolution values to be positive. "
                        f"{axis_name=} {value=}"
                    )
                else:
                    assert value > 0, (
                        "Expected render resolution values to be positive integers. "
                        f"{axis_name=} {value=} {resolution=}"
                    )
        assert camera_name is None or isinstance(camera_name, str), (
            "Expected `camera_name` to be None or a string. " f"{type(camera_name)=}"
        )
        if display_cameras is not None:
            assert isinstance(display_cameras, list), (
                "Expected `display_cameras` to be a list when provided. "
                f"{type(display_cameras)=}"
            )
            for display_camera in display_cameras:
                assert isinstance(display_camera, Camera), (
                    "Expected every display camera to be a Camera. "
                    f"{type(display_camera)=}"
                )
        assert title is None or isinstance(title, str), (
            "Expected `title` to be None or a string. " f"{type(title)=}"
        )
        assert device is None or isinstance(device, torch.device), (
            "Expected `device` to be None or a torch.device. " f"{type(device)=}"
        )

    _validate_inputs()

    def _normalize_inputs(
        camera: Camera,
        resolution: Optional[Tuple[int, int]],
        title: Optional[str],
        device: Optional[torch.device],
    ) -> Tuple[Camera, Tuple[int, int], str, torch.device]:
        camera = camera.to(device if device is not None else scene_model.device)
        if resolution is None:
            resolution = camera.intrinsics.resolution
        normalized = []
        for value in resolution:
            if isinstance(value, torch.Tensor):
                normalized.append(int(value.detach().cpu().item()))
            else:
                normalized.append(value)
        if title is None:
            title = ""
        if device is None:
            device = scene_model.device
        return camera, (normalized[0], normalized[1]), title, device

    camera, resolution, title, device = _normalize_inputs(
        camera=camera,
        resolution=resolution,
        title=title,
        device=device,
    )

    image: Optional[torch.Tensor] = None
    if camera_name is not None:
        image = scene_model._get_snapshot(camera_name)

    if image is None:
        image = render_rgb_from_mesh_func(
            mesh=scene_model.model,
            camera=camera,
            resolution=resolution,
        )
        if camera_name is not None:
            snapshot = image.detach().cpu()
            scene_model._put_snapshot(camera_name, snapshot)

    composed = BaseSceneModel._apply_camera_overlays(
        image=image,
        display_cameras=display_cameras,
        render_at_camera=camera,
        resolution=resolution,
    )

    display_payload = {
        'image': composed,
        'title': title,
    }
    return display_payload
