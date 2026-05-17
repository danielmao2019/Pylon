"""Camera visualization primitives."""

from typing import Any, Dict, List, Optional

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras


def camera_vis(
    camera: Camera,
    axis_length: float = 4.0,
    frustum_depth: float = 8.0,
    frustum_color: torch.Tensor | None = None,
) -> Dict[str, Any]:
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert axis_length > 0.0
    assert frustum_depth > 0.0

    device = camera.device
    dtype = camera.center.dtype
    if frustum_color is None:
        frustum_color = torch.tensor([1.0, 0.84, 0.0], device=device, dtype=dtype)

    axes = [
        {
            'start': camera.center,
            'end': camera.center + camera.right * axis_length,
            'color': torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        },
        {
            'start': camera.center,
            'end': camera.center + camera.forward * axis_length,
            'color': torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        },
        {
            'start': camera.center,
            'end': camera.center + camera.up * axis_length,
            'color': torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        },
    ]

    frustum_half_size = frustum_depth * 0.5
    frustum_center = camera.center + camera.forward * frustum_depth
    frustum_points_world = [
        frustum_center
        - camera.right * frustum_half_size
        + camera.up * frustum_half_size,
        frustum_center
        + camera.right * frustum_half_size
        + camera.up * frustum_half_size,
        frustum_center
        + camera.right * frustum_half_size
        - camera.up * frustum_half_size,
        frustum_center
        - camera.right * frustum_half_size
        - camera.up * frustum_half_size,
    ]

    frustum_lines: List[Dict[str, Any]] = []
    for point in frustum_points_world:
        frustum_lines.append(
            {
                'start': camera.center,
                'end': point,
                'color': frustum_color,
            }
        )

    for idx in range(len(frustum_points_world)):
        frustum_lines.append(
            {
                'start': frustum_points_world[idx],
                'end': frustum_points_world[(idx + 1) % len(frustum_points_world)],
                'color': frustum_color,
            }
        )

    return {
        'center': camera.center,
        'axes': axes,
        'frustum_lines': frustum_lines,
        'center_color': torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype),
    }


def cameras_vis(
    cameras: Cameras,
    frustum_scale: float,
    frustum_color: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    """Build camera visualization primitives for a camera collection.

    Args:
        cameras: Camera collection to visualize.
        frustum_scale: Frustum depth in world units.
        frustum_color: Optional RGB tensor for frustum line color.

    Returns:
        Camera visualization primitive dictionaries.
    """
    assert isinstance(cameras, Cameras), (
        "Cameras must be a Cameras collection. cameras=%r" % cameras
    )
    assert isinstance(frustum_scale, float), (
        "Frustum scale must be a float. frustum_scale=%r" % frustum_scale
    )
    assert frustum_scale > 0.0, (
        "Frustum scale must be positive. frustum_scale=%r" % frustum_scale
    )
    assert frustum_color is None or isinstance(frustum_color, torch.Tensor), (
        "Frustum color must be None or a tensor. frustum_color=%r" % frustum_color
    )

    camera_visualizations: List[Dict[str, Any]] = []
    for camera in cameras:
        camera_visualizations.append(
            camera_vis(
                camera=camera,
                axis_length=frustum_scale * 0.6,
                frustum_depth=frustum_scale,
                frustum_color=frustum_color,
            )
        )
    return camera_visualizations
