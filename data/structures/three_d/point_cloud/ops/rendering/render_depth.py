"""Depth rendering from point clouds using projection methods."""

from typing import Tuple, Union

import torch
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.ops.rendering.common import (
    prepare_points_for_rendering,
    validate_rendering_inputs,
)
from data.structures.three_d.point_cloud.ops.rendering.render_mask import (
    render_mask_from_rendering_points,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def render_depth_from_rendering_points(
    rendering_points: torch.Tensor,
    resolution: Tuple[int, int],
    ignore_value: float = float('inf'),
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render depth map from pre-processed rendered points.

    Args:
        rendering_points: Pre-processed points [M, 3] with (x, y, depth).
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: inf).
        return_mask: If True, also return valid pixel mask (default: False).

    Returns:
        If return_mask is False:
            Depth map tensor of shape [H, W] with depth values.
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [H, W]).
    """
    render_height, render_width = resolution

    # Allocate depth map
    depth_map = torch.full(
        (render_height, render_width),
        ignore_value,
        dtype=torch.float32,
        device=rendering_points.device,
    )

    # Render pixels
    depth_map[rendering_points[:, 1].long(), rendering_points[:, 0].long()] = (
        rendering_points[:, 2].float()
    )

    # Handle mask creation if requested
    if return_mask:
        valid_mask = render_mask_from_rendering_points(
            rendering_points=rendering_points,
            resolution=resolution,
            device=rendering_points.device,
        )
        return depth_map, valid_mask
    else:
        return depth_map


def render_depth_from_point_cloud(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: float = -1.0,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render depth map from point cloud using camera projection.

    Projects 3D point cloud coordinates onto 2D image plane using camera
    parameters and generates a depth map. Supports circular point rendering
    for improved visualization.

    Args:
        pc: Point cloud data containing xyz coordinates.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: -1.0).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            Depth map tensor of shape [H, W] with depth values in camera coordinate system.
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [H, W]).

    Raises:
        AssertionError: If point cloud is empty or no points project within image bounds.
        NotImplementedError: If convention other than "opengl" is specified.
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"

    # Validate inputs
    validate_rendering_inputs(
        pc=pc,
        camera=camera,
        resolution=resolution,
        ignore_value=ignore_value,
        return_mask=return_mask,
        point_size=point_size,
    )

    # Prepare points for rendering
    rendered_points, _ = prepare_points_for_rendering(
        pc=pc,
        camera=camera,
        resolution=resolution,
    )

    # Render depth map
    return render_depth_from_rendering_points(
        rendering_points=rendered_points,
        resolution=resolution,
        ignore_value=ignore_value,
        return_mask=return_mask,
    )
