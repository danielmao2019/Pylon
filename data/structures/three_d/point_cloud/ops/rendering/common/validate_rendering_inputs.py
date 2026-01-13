from typing import Optional, Tuple, Union

import torch
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def validate_rendering_inputs(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: Optional[Union[int, float]] = None,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> None:
    """Validate common inputs for point cloud rendering operations.

    Args:
        pc: Point cloud data containing xyz coordinates
        camera: Camera object containing intrinsics, extrinsics, and convention
        resolution: Target resolution as (height, width) tuple
        ignore_value: Optional ignore value to validate (if provided)
        return_mask: Whether to return a mask along with the rendered output
        point_size: Size of rendered points in pixels (default: 1.0)

    Raises:
        AssertionError: If validation fails
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    points = pc.xyz

    intrinsics = camera.intrinsics
    extrinsics = camera.extrinsics
    assert intrinsics.device == points.device, (
        f"points device {points.device} != camera_intrinsics device {intrinsics.device}"
    )
    assert extrinsics.device == points.device, (
        f"points device {points.device} != camera_extrinsics device {extrinsics.device}"
    )

    # Validate resolution and convention
    assert isinstance(
        resolution, (tuple, list)
    ), f"resolution must be tuple or list, got {type(resolution)}"
    assert (
        len(resolution) == 2
    ), f"resolution must have 2 elements (height, width), got {len(resolution)}"
    assert all(
        isinstance(x, int) and x > 0 for x in resolution
    ), f"resolution must be positive integers, got {resolution}"

    # Validate ignore_value if provided
    if ignore_value is not None:
        assert isinstance(
            ignore_value, (int, float)
        ), f"ignore_value must be int or float, got {type(ignore_value)}"

    # Validate return_mask
    assert isinstance(
        return_mask, bool
    ), f"return_mask must be bool, got {type(return_mask)}"

    # Validate point_size
    assert isinstance(
        point_size, (int, float)
    ), f"point_size must be numeric, got {type(point_size)}"
    assert point_size >= 1.0, f"point_size must be >= 1.0, got {point_size}"
