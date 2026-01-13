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
    dtype = camera.intrinsics.dtype

    extrinsics_standard = camera.to(device=device, convention='standard').extrinsics
    rotation_matrix = extrinsics_standard[:3, :3]
    camera_center = extrinsics_standard[:3, 3]

    axis_colors = {
        'x': torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        'y': torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        'z': torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
    }
    if frustum_color is None:
        frustum_color = torch.tensor([1.0, 0.84, 0.0], device=device, dtype=dtype)

    right_axis = rotation_matrix[:, 0] * axis_length
    forward_axis = rotation_matrix[:, 1] * axis_length
    up_axis = rotation_matrix[:, 2] * axis_length

    axes = [
        {
            'start': camera_center,
            'end': camera_center + right_axis,
            'color': axis_colors['x'],
        },
        {
            'start': camera_center,
            'end': camera_center + forward_axis,
            'color': axis_colors['y'],
        },
        {
            'start': camera_center,
            'end': camera_center + up_axis,
            'color': axis_colors['z'],
        },
    ]

    frustum_points_cam = (
        torch.tensor(
            [
                [-0.5, 1.0, -0.5],
                [0.5, 1.0, -0.5],
                [0.5, 1.0, 0.5],
                [-0.5, 1.0, 0.5],
            ],
            device=device,
            dtype=dtype,
        )
        * frustum_depth
    )

    frustum_points_cam_homo = torch.cat(
        [
            frustum_points_cam,
            torch.ones(frustum_points_cam.shape[0], 1, device=device, dtype=dtype),
        ],
        dim=1,
    )
    frustum_points_world = (extrinsics_standard @ frustum_points_cam_homo.t()).t()[
        :, :3
    ]

    frustum_lines: List[Dict[str, Any]] = []
    origin = camera_center
    for point in frustum_points_world:
        frustum_lines.append(
            {
                'start': origin,
                'end': point,
                'color': frustum_color,
            }
        )

    for idx in range(4):
        frustum_lines.append(
            {
                'start': frustum_points_world[idx],
                'end': frustum_points_world[(idx + 1) % 4],
                'color': frustum_color,
            }
        )

    return {
        'center': camera_center,
        'axes': axes,
        'frustum_lines': frustum_lines,
        'center_color': torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype),
    }
