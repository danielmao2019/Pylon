"""Depth rendering from point clouds using projection methods."""

from typing import Optional, Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import (
    apply_point_size_postprocessing,
)
from models.three_d.point_cloud.render.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from models.three_d.point_cloud.render.common.select_nearest_point_per_pixel import (
    select_nearest_point_per_pixel,
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
    depth_map = render_depth_from_rendering_points(
        rendering_points=rendered_points,
        resolution=resolution,
        ignore_value=ignore_value,
        return_mask=False,
        valid=valid,
    )

    # Dilate each rendered point into a disc of point_size pixels
    if point_size > 1.0:
        depth_map = apply_point_size_postprocessing(
            rendered_image=depth_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=ignore_value,
        )

    if return_mask:
        if point_size > 1.0:
            # The dilation repainted the depth map, so the mask follows it
            valid_mask = depth_map != ignore_value
        else:
            valid_mask = render_mask_from_rendering_points(
                rendering_points=rendered_points,
                resolution=resolution,
                device=rendered_points.device,
                valid=valid,
            )

        return depth_map, valid_mask
    else:
        return depth_map


def render_depth_from_rendering_points(
    rendering_points: torch.Tensor,
    resolution: Tuple[int, int],
    ignore_value: float = float('inf'),
    return_mask: bool = False,
    valid: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Rasterize already-projected points into a depth map.

    Reads at each pixel the depth of the point that owns it, so occlusion is
    decided by depth rather than by which write landed last.

    Args:
        rendering_points: Pre-processed points [..., N, 3] with (x, y, depth), the
            point axis in pc.xyz order and the leading axes enumerating the
            cameras rendered.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: inf).
        return_mask: If True, also return valid pixel mask (default: False).
        valid: Optional [..., N] bool torch.Tensor marking which points each
            camera keeps; a point marked False never owns a pixel, and None
            means every point of rendering_points is marked.

    Returns:
        If return_mask is False:
            Depth map tensor of shape [..., H, W] with depth values.
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [..., H, W]).
    """
    valid = (
        torch.ones(
            rendering_points.shape[:-1],
            dtype=torch.bool,
            device=rendering_points.device,
        )
        if valid is None
        else valid
    )

    # Resolve which point owns each pixel, then read that point's own depth
    winner = select_nearest_point_per_pixel(
        rendering_points=rendering_points,
        valid=valid,
        resolution=resolution,
    )
    depth_map = torch.gather(
        rendering_points[..., 2],
        dim=-1,
        index=winner.clamp(min=0).reshape(winner.shape[:-2] + (-1,)),
    ).reshape(winner.shape)
    depth_map = depth_map.float().masked_fill(winner < 0, ignore_value)

    # Handle mask creation if requested
    if return_mask:
        valid_mask = render_mask_from_rendering_points(
            rendering_points=rendering_points,
            resolution=resolution,
            device=rendering_points.device,
            valid=valid,
        )
        return depth_map, valid_mask
    else:
        return depth_map
