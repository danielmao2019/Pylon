"""Mask rendering from point clouds using projection methods."""

from typing import Optional, Tuple

import torch


def render_mask_from_rendering_points(
    rendering_points: torch.Tensor,
    resolution: Tuple[int, int],
    device: torch.device,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mark the pixels a surviving point landed on.

    Args:
        rendering_points: Pre-processed points [..., N, 3] float torch.Tensor of (x, y, depth), the point axis in pc.xyz order and the leading axes enumerating the cameras rendered.
        resolution: Target resolution as (height, width) tuple.
        device: Device for the tensor.
        valid: Optional [..., N] bool torch.Tensor marking which points each camera keeps; None means every point of rendering_points is marked.

    Returns:
        Boolean mask torch.Tensor of shape [..., H, W] indicating valid pixels, carrying the leading axes of rendering_points.
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

    valid_mask = (nearest_point_index >= 0).to(device)

    return valid_mask
