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
        rendering_points: Pre-processed points [..., N, 3] float torch.Tensor of (x, y, depth), the leading axes enumerating the cameras rendered.
        resolution: Target resolution as (height, width) tuple.
        device: Device for the tensor.
        valid: Optional [..., N] bool torch.Tensor marking which points each camera keeps, the only points the rasterization reads; None means the caller already dropped its culled points, so every point of rendering_points is read.

    Returns:
        Boolean mask torch.Tensor of shape [..., H, W] indicating valid pixels, carrying the leading axes of rendering_points.
    """
    render_height, render_width = resolution
    num_points = rendering_points.shape[-2]

    # Each kept entry is a (camera, point) pair, flattened as camera * num_points + point
    if valid is None:
        # The caller already dropped its culled points, as a single camera does
        kept = torch.arange(
            rendering_points.shape[:-1].numel(), device=rendering_points.device
        )
    else:
        # The reduction reads these alone, so no work goes to the points a camera culled
        kept = torch.nonzero(valid.reshape(-1), as_tuple=True)[0]
    kept_points = rendering_points.reshape(-1, 3)[kept]
    kept_pixel = (
        (kept // num_points) * (render_height * render_width)
        + kept_points[:, 1].long() * render_width
        + kept_points[:, 0].long()
    )
    kept_depth = kept_points[:, 2]

    # Resolve, per pixel, the kept point with the smallest depth landing there, reduced per pixel rather than scattered so occlusion does not depend on which write lands last
    num_pixels = rendering_points.shape[:-2].numel() * render_height * render_width
    nearest_depth = torch.full(
        size=(num_pixels,),
        fill_value=float('inf'),
        dtype=rendering_points.dtype,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=0,
        index=kept_pixel,
        src=kept_depth,
        reduce='amin',
        include_self=True,
    )
    # The point indices reduce the same way, so two points tying on depth resolve to the lower index.
    owning = torch.nonzero(kept_depth == nearest_depth[kept_pixel], as_tuple=True)[0]
    nearest_point_index = torch.full(
        size=(num_pixels,),
        fill_value=num_points,
        dtype=torch.int64,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=0,
        index=kept_pixel[owning],
        src=kept[owning] % num_points,
        reduce='amin',
        include_self=True,
    )
    nearest_point_index = nearest_point_index.masked_fill(
        nearest_point_index == num_points, -1
    ).reshape(rendering_points.shape[:-2] + (render_height, render_width))

    valid_mask = (nearest_point_index >= 0).to(device)

    return valid_mask
