"""Camera visualization primitives."""

from typing import Any, Dict, List

import torch

from data.structures.three_d.camera.camera import Camera


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
