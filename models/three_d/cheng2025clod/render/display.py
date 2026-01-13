from typing import Any, Dict, List, Optional, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.cheng2025clod.render.core import render_rgb_from_cheng2025_clod


def render_display(
    scene_model: BaseSceneModel,
    camera: Camera,
    resolution: Tuple[int, int],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    device: Optional[torch.device],
) -> Dict[str, Any]:
    # Input validations
    assert isinstance(scene_model, BaseSceneModel), f"{type(scene_model)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert (
        isinstance(resolution, tuple)
        and len(resolution) == 2
        and all(isinstance(v, int) for v in resolution)
    ), f"{resolution=}"
    if camera_name is not None:
        assert isinstance(camera_name, str), f"{type(camera_name)=}"
    if display_cameras is not None:
        assert isinstance(display_cameras, list), f"{type(display_cameras)=}"
        assert all(isinstance(cam, Camera) for cam in display_cameras)
    if title is not None:
        assert isinstance(title, str), f"{type(title)=}"
    if device is not None:
        assert isinstance(device, torch.device), f"{type(device)=}"
    assert hasattr(scene_model, '_virtual_scale'), "scene_model missing _virtual_scale"
    assert hasattr(scene_model, '_lod_threshold'), "scene_model missing _lod_threshold"
    assert scene_model._virtual_scale is not None, "_virtual_scale must be set"
    assert scene_model._lod_threshold is not None, "_lod_threshold must be set"

    resolved_device = device if device is not None else scene_model.device
    camera = camera.to(resolved_device)

    image: Optional[torch.Tensor] = None
    if camera_name is not None:
        image = scene_model._get_snapshot(camera_name)

    if image is None:
        image = render_rgb_from_cheng2025_clod(
            model=scene_model.model,
            camera=camera,
            resolution=resolution,
            virtual_scale=scene_model._virtual_scale,
            lod_threshold=scene_model._lod_threshold,
            device=resolved_device,
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
    title_value = title if title is not None else ""
    return {
        "image": composed,
        "title": title_value,
    }
