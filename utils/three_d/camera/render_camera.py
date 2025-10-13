"""Render camera geometry into image space using Bresenham lines."""

from typing import Tuple, Union
import torch
from utils.three_d.camera.camera_vis import camera_vis
from utils.three_d.camera.conventions import apply_coordinate_transform
from utils.three_d.camera.project import project_3d_to_2d
from utils.three_d.camera.transform import world_to_camera_transform


def render_camera(
    cam_intrinsics: torch.Tensor,
    cam_extrinsics: torch.Tensor,
    cam_convention: str,
    render_at_intrinsics: torch.Tensor,
    render_at_extrinsics: torch.Tensor,
    render_at_resolution: Tuple[int, int],
    render_at_convention: str,
    return_mask: bool = False,
    axis_length: float = 4.0,
    frustum_depth: float = 8.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    assert isinstance(cam_intrinsics, torch.Tensor)
    assert cam_intrinsics.shape == (3, 3)
    assert isinstance(cam_extrinsics, torch.Tensor)
    assert cam_extrinsics.shape == (4, 4)
    assert isinstance(render_at_intrinsics, torch.Tensor)
    assert render_at_intrinsics.shape == (3, 3)
    assert isinstance(render_at_extrinsics, torch.Tensor)
    assert render_at_extrinsics.shape == (4, 4)
    height, width = render_at_resolution
    assert height > 0
    assert width > 0

    device = render_at_intrinsics.device
    dtype = render_at_intrinsics.dtype

    geometry = camera_vis(
        intrinsics=cam_intrinsics.to(device=device, dtype=dtype),
        extrinsics=cam_extrinsics.to(device=device, dtype=dtype),
        convention=cam_convention,
        axis_length=axis_length,
        frustum_depth=frustum_depth,
    )

    render_intrinsics = render_at_intrinsics.to(device=device, dtype=dtype)
    render_extrinsics_opencv = apply_coordinate_transform(
        extrinsics=render_at_extrinsics.to(device=device, dtype=dtype),
        source_convention=render_at_convention,
        target_convention='opencv',
    )

    overlay = torch.zeros(3, height, width, device=device, dtype=dtype)
    mask = torch.zeros(height, width, device=device, dtype=dtype)

    def project(points: torch.Tensor) -> torch.Tensor | None:
        points_cam = world_to_camera_transform(
            points=points.to(device=device, dtype=dtype),
            extrinsics=render_extrinsics_opencv,
            inplace=False,
        )
        if not torch.all(points_cam[:, 2] > 1.0e-4):
            return None
        pixels = project_3d_to_2d(
            points=points_cam,
            intrinsics=render_intrinsics,
            inplace=False,
        )
        in_bounds = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        if not torch.all(in_bounds):
            return None
        return torch.round(pixels).long()

    def draw_segment(
        start: torch.Tensor, end: torch.Tensor, color: torch.Tensor
    ) -> None:
        pixels = project(torch.stack([start, end], dim=0))
        if pixels is None or pixels.shape[0] < 2:
            return
        draw_color = color.to(device=device, dtype=dtype)
        u0 = float(pixels[0, 0].item())
        v0 = float(pixels[0, 1].item())
        u1 = float(pixels[1, 0].item())
        v1 = float(pixels[1, 1].item())
        du = u1 - u0
        dv = v1 - v0
        steps = int(max(abs(du), abs(dv))) + 1
        t = torch.linspace(0.0, 1.0, steps, device=device, dtype=dtype)
        u = torch.round(u0 + t * du).long()
        v = torch.round(v0 + t * dv).long()
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not torch.any(valid):
            return
        u = u[valid]
        v = v[valid]
        overlay[:, v, u] = draw_color.view(3, 1).expand(-1, u.shape[0])
        mask[v, u] = 1.0

    segments = geometry['axes'] + geometry['frustum_lines']
    for segment in segments:
        draw_segment(segment['start'], segment['end'], segment['color'])

    center_proj = project(geometry['center'].unsqueeze(0))
    if center_proj is not None and center_proj.shape[0] > 0:
        cu = int(center_proj[0, 0].item())
        cv = int(center_proj[0, 1].item())
        assert 0 <= cu < width and 0 <= cv < height
        overlay[:, cv, cu] = geometry['center_color'].to(device=device, dtype=dtype)
        mask[cv, cu] = 1.0

    return (overlay, mask) if return_mask else overlay
