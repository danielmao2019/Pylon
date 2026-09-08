"""Depth rendering from point clouds using projection methods."""

from typing import Optional, Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from models.three_d.point_cloud.render.common.validate_rendering_inputs import (
    validate_rendering_inputs,
)
from models.three_d.point_cloud.render.render_mask import (
    render_mask_from_rendering_points,
)


def render_depth_from_point_cloud(
    pc: PointCloud,
    camera: Union[Camera, Cameras],
    resolution: Tuple[int, int],
    ignore_value: float = -1.0,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render depth map from point cloud using camera projection.

    Projects 3D point cloud coordinates onto the 2D image plane using camera
    parameters and generates a depth map, chaining validation, projection, and
    rasterization; a Camera gives [H, W] and a Cameras gives [B, H, W] down the
    same path.

    Args:
        pc: Point cloud data containing xyz coordinates.
        camera: The Camera (no leading axis) or Cameras (a [B] leading axis)
            containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: -1.0).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            Depth map tensor of shape [..., H, W] with depth values in the camera
            coordinate system, carrying the camera's leading axes.
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [..., H, W]).

    Raises:
        AssertionError: If point cloud is empty or no points project within image bounds.
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
    rendered_points, valid = prepare_points_for_rendering(
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
        valid=valid,
    )


def render_depth_from_rendering_points(
    rendering_points: torch.Tensor,
    resolution: Tuple[int, int],
    ignore_value: float = float('inf'),
    return_mask: bool = False,
    valid: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render depth map from pre-processed rendered points.

    Args:
        rendering_points: Pre-processed points [..., N, 3] with (x, y, depth), the
            leading axes enumerating the cameras rendered.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: inf).
        return_mask: If True, also return valid pixel mask (default: False).
        valid: Optional [..., N] bool torch.Tensor marking which points each
            camera keeps; None means every point of rendering_points is written.

    Returns:
        If return_mask is False:
            Depth map tensor of shape [..., H, W] with depth values.
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [..., H, W]).
    """

    def _normalize_inputs(valid: Optional[torch.Tensor]) -> torch.Tensor:
        if valid is None:
            valid = torch.ones(
                rendering_points.shape[:-1],
                dtype=torch.bool,
                device=rendering_points.device,
            )
        return valid

    valid = _normalize_inputs(valid=valid)

    render_height, render_width = resolution

    # Allocate depth map, its leading axes those of rendering_points
    depth_map = torch.full(
        rendering_points.shape[:-2] + (render_height, render_width),
        ignore_value,
        dtype=torch.float32,
        device=rendering_points.device,
    )

    # Render pixels, each index selected by valid so a culled point writes nowhere
    selector = torch.nonzero(valid, as_tuple=True)
    depth_map[
        selector[:-1]
        + (
            rendering_points[..., 1][selector].long(),
            rendering_points[..., 0][selector].long(),
        )
    ] = rendering_points[..., 2][selector].float()

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
