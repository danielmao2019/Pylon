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

    Projects 3D point cloud coordinates onto the 2D image plane using camera parameters and generates a depth map, chaining validation, projection, and rasterization; a Camera gives [H, W] and a Cameras gives [B, H, W] down the same path.

    Args:
        pc: Point cloud data containing xyz coordinates.
        camera: The Camera (no leading axis) or Cameras (a [B] leading axis) containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: -1.0).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            Depth map tensor of shape [..., H, W] with depth values in the camera coordinate system, carrying the camera's leading axes.
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

    if point_size > 1.0:
        # Render depth map, positive infinity wherever no point landed
        depth_map = render_depth_from_rendering_points(
            rendering_points=rendered_points,
            resolution=resolution,
            ignore_value=float('inf'),
            return_mask=False,
            valid=valid,
        )

        # Dilate each rendered point into a disc of point_size pixels
        depth_map = apply_point_size_postprocessing(
            rendered_image=depth_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )

        # The discs the dilation reached, read off the infinity sentinel rather than ignore_value, which may be NaN
        covered = torch.isfinite(depth_map)
        depth_map = depth_map.masked_fill(~covered, ignore_value)
    else:
        # Render depth map
        depth_map = render_depth_from_rendering_points(
            rendering_points=rendered_points,
            resolution=resolution,
            ignore_value=ignore_value,
            return_mask=False,
            valid=valid,
        )

    if return_mask:
        if point_size > 1.0:
            # The coverage the map's own dilation reached, so the mask and the map it describes cannot drift apart
            valid_mask = covered
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

    Reads at each pixel the depth of the point that owns it, so occlusion is decided by depth rather than by which write landed last.

    Args:
        rendering_points: Pre-processed points [..., N, 3] with (x, y, depth), the point axis in pc.xyz order and the leading axes enumerating the cameras rendered.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: inf).
        return_mask: If True, also return valid pixel mask (default: False).
        valid: Optional [..., N] bool torch.Tensor marking which points each camera keeps; a point marked False never owns a pixel, and None means every point of rendering_points is marked.

    Returns:
        If return_mask is False:
            Depth map tensor of shape [..., H, W] with depth values.
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [..., H, W]).
    """
    render_height, render_width = resolution
    if valid is None:
        valid = torch.ones(
            rendering_points.shape[:-1],
            dtype=torch.bool,
            device=rendering_points.device,
        )

    # Resolve, per pixel, the valid point with the smallest depth landing there, reduced per pixel rather than scattered so occlusion does not depend on which write lands last. A culled point is parked on pixel 0, whose out-of-image coordinates are not scatterable, and its depth of positive infinity keeps it from ever owning that pixel.
    num_points = rendering_points.shape[-2]
    pixel_index = (
        rendering_points[..., 1].long() * render_width + rendering_points[..., 0].long()
    )
    pixel_index = pixel_index.masked_fill(~valid, 0)
    depth_key = rendering_points[..., 2].masked_fill(~valid, float('inf'))
    nearest_depth = torch.full(
        size=rendering_points.shape[:-2] + (render_height * render_width,),
        fill_value=float('inf'),
        dtype=rendering_points.dtype,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=-1,
        index=pixel_index,
        src=depth_key,
        reduce='amin',
        include_self=True,
    )
    # The point indices reduce the same way, so two points tying on depth resolve to the lower index.
    point_index = torch.arange(
        num_points, dtype=torch.int64, device=rendering_points.device
    ).expand_as(pixel_index)
    owns_pixel = valid & (depth_key == nearest_depth.gather(dim=-1, index=pixel_index))
    nearest_point_index = torch.full(
        size=rendering_points.shape[:-2] + (render_height * render_width,),
        fill_value=num_points,
        dtype=torch.int64,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=-1,
        index=pixel_index,
        src=torch.where(owns_pixel, point_index, num_points),
        reduce='amin',
        include_self=True,
    )
    nearest_point_index = nearest_point_index.masked_fill(
        nearest_point_index == num_points, -1
    ).reshape(rendering_points.shape[:-2] + (render_height, render_width))

    # Read the depth of the point that owns each pixel
    depth_map = torch.gather(
        rendering_points[..., 2],
        dim=-1,
        index=nearest_point_index.clamp(min=0).reshape(
            nearest_point_index.shape[:-2] + (-1,)
        ),
    ).reshape(nearest_point_index.shape)
    depth_map = depth_map.float().masked_fill(nearest_point_index < 0, ignore_value)

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
