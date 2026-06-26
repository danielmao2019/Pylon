"""Camera visualization primitives."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras

DEFAULT_FRUSTUM_SIZE = 0.25
DEFAULT_FRUSTUM_COLOR = (255, 214, 0)
DEFAULT_POINT_SIZE = 0.01
DEFAULT_POINT_COLOR = (255, 214, 0)


def cameras_vis(
    cameras: Cameras,
    frustum_size: Optional[float] = None,
    frustum_color: Optional[Tuple[int, int, int]] = None,
    point_size: Optional[float] = None,
    point_color: Optional[Tuple[int, int, int]] = None,
) -> List[Dict[str, Any]]:
    """Build per-camera visualization primitives for a camera collection.

    Args:
        cameras: Camera collection to visualize.
        frustum_size: World-unit frustum/axis size, forwarded untouched; `None` is
            resolved to the per-camera default by `camera_vis`.
        frustum_color: RGB line color as a 0-255 int tuple, forwarded untouched;
            `None` is resolved to the per-camera default by `camera_vis`.
        point_size: World-unit camera-center marker size, forwarded untouched;
            `None` is resolved to the per-camera default by `camera_vis`.
        point_color: RGB center-point color as a 0-255 int tuple, forwarded
            untouched; `None` is resolved to the per-camera default by `camera_vis`.

    Returns:
        List of per-camera visualization primitive dicts, one per camera, each with
        keys `center`, `center_color`, `center_size`, `axes`, and `frustum_lines`.
    """
    assert isinstance(cameras, Cameras), (
        "Cameras must be a Cameras collection. " f"{type(cameras)=}"
    )
    assert frustum_size is None or isinstance(frustum_size, float), (
        "Frustum size must be None or a float. " f"{frustum_size=}"
    )
    assert frustum_color is None or isinstance(frustum_color, tuple), (
        "Frustum color must be None or a tuple. " f"{frustum_color=}"
    )
    assert point_size is None or isinstance(point_size, float), (
        "Point size must be None or a float. " f"{point_size=}"
    )
    assert point_color is None or isinstance(point_color, tuple), (
        "Point color must be None or a tuple. " f"{point_color=}"
    )

    camera_visualizations: List[Dict[str, Any]] = []
    for camera in cameras:
        camera_visualizations.append(
            camera_vis(
                camera=camera,
                frustum_size=frustum_size,
                frustum_color=frustum_color,
                point_size=point_size,
                point_color=point_color,
            )
        )
    return camera_visualizations


def camera_vis(
    camera: Camera,
    frustum_size: Optional[float] = None,
    frustum_color: Optional[Tuple[int, int, int]] = None,
    point_size: Optional[float] = None,
    point_color: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """Build the visualization primitive for a single camera.

    Args:
        camera: Camera whose center and right/forward/up basis define the geometry.
        frustum_size: World-unit frustum/axis size; `None` resolves to
            `DEFAULT_FRUSTUM_SIZE`. The axes use length `0.6 * frustum_size` and the
            frustum depth is `frustum_size`.
        frustum_color: RGB frustum-line color as a 0-255 int tuple; `None` resolves
            to `DEFAULT_FRUSTUM_COLOR`.
        point_size: World-unit camera-center marker size; `None` resolves to
            `DEFAULT_POINT_SIZE`.
        point_color: RGB center-point color as a 0-255 int tuple; `None` resolves to
            `DEFAULT_POINT_COLOR`.

    Returns:
        Dict with keys `center` (the camera-center `float [3]` tensor),
        `center_color` (a normalized float RGB `[3]` tensor), `center_size` (a plain
        Python float), `axes` (a length-3 list of line dicts), and `frustum_lines`
        (a length-8 list of line dicts). Each line dict has keys `start`, `end`, and
        `color`, all float tensors on `camera.device` with `camera.center.dtype`.
    """
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert frustum_size is None or isinstance(frustum_size, float), (
        "Frustum size must be None or a float. " f"{frustum_size=}"
    )
    assert frustum_color is None or isinstance(frustum_color, tuple), (
        "Frustum color must be None or a tuple. " f"{frustum_color=}"
    )
    assert point_size is None or isinstance(point_size, float), (
        "Point size must be None or a float. " f"{point_size=}"
    )
    assert point_color is None or isinstance(point_color, tuple), (
        "Point color must be None or a tuple. " f"{point_color=}"
    )

    # --- Resolve defaults ---
    if frustum_size is None:
        frustum_size = DEFAULT_FRUSTUM_SIZE
    if frustum_color is None:
        frustum_color = DEFAULT_FRUSTUM_COLOR
    if point_size is None:
        point_size = DEFAULT_POINT_SIZE
    if point_color is None:
        point_color = DEFAULT_POINT_COLOR

    extrinsics = camera.extrinsics
    intrinsics = camera.intrinsics
    device = camera.device
    center = extrinsics.center
    right = extrinsics.right
    forward = extrinsics.forward
    up = extrinsics.up
    dtype = center.dtype

    def _to_color_tensor(color: Tuple[int, int, int]) -> torch.Tensor:
        return torch.tensor(
            [channel / 255.0 for channel in color],
            device=device,
            dtype=dtype,
        )

    # --- Center marker ---
    center_color = _to_color_tensor(color=point_color)
    center_size = float(point_size)

    # --- Axes (right=red, forward=green, up=blue) ---
    axis_length = 0.6 * frustum_size
    axes = [
        {
            'start': center,
            'end': center + right * axis_length,
            'color': torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        },
        {
            'start': center,
            'end': center + forward * axis_length,
            'color': torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        },
        {
            'start': center,
            'end': center + up * axis_length,
            'color': torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        },
    ]

    # --- Frustum ---
    frustum_depth = frustum_size
    frustum_line_color = _to_color_tensor(color=frustum_color)
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy
    if cx > 0.0 and cy > 0.0 and fx > 0.0 and fy > 0.0:
        half_width = frustum_depth * (cx / fx)
        half_height = frustum_depth * (cy / fy)
    else:
        half_width = frustum_depth * 0.5
        half_height = frustum_depth * 0.5
    frustum_center = center + forward * frustum_depth
    frustum_points_world = [
        frustum_center - right * half_width + up * half_height,
        frustum_center + right * half_width + up * half_height,
        frustum_center + right * half_width - up * half_height,
        frustum_center - right * half_width - up * half_height,
    ]

    frustum_lines: List[Dict[str, Any]] = []
    for point in frustum_points_world:
        frustum_lines.append(
            {
                'start': center,
                'end': point,
                'color': frustum_line_color,
            }
        )
    for idx in range(len(frustum_points_world)):
        frustum_lines.append(
            {
                'start': frustum_points_world[idx],
                'end': frustum_points_world[(idx + 1) % len(frustum_points_world)],
                'color': frustum_line_color,
            }
        )

    return {
        'center': center,
        'center_color': center_color,
        'center_size': center_size,
        'axes': axes,
        'frustum_lines': frustum_lines,
    }
