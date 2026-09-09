from typing import Tuple

import torch


def select_nearest_point_per_pixel(
    rendering_points: torch.Tensor,
    valid: torch.Tensor,
    resolution: Tuple[int, int],
) -> torch.Tensor:
    """Resolve, per pixel, which surviving point a rasterizer writes there.

    Args:
        rendering_points: [..., N, 3] float torch.Tensor of already-projected
            points as (x, y, depth), x measured against the width and y against
            the height, the point axis in pc.xyz order and the leading axes
            enumerating the cameras rendered.
        valid: [..., N] bool torch.Tensor marking which points each camera keeps;
            a point marked False never owns a pixel.
        resolution: Target image resolution as an (H, W) tuple.

    Returns:
        [..., H, W] int64 torch.Tensor carrying at each pixel the index, along the
        point axis of rendering_points, of the surviving point nearest the camera,
        and -1 at each pixel no surviving point landed on.
    """
    render_height, render_width = resolution
    num_points = rendering_points.shape[-2]

    # Pack each point's pixel into one flat index. A culled point is parked on
    # pixel 0, whose out-of-image coordinates are not scatterable, and the depth
    # of positive infinity below keeps it from ever winning that pixel.
    pixel_index = (
        rendering_points[..., 1].long() * render_width + rendering_points[..., 0].long()
    )
    pixel_index = pixel_index.masked_fill(~valid, 0)
    depth_key = rendering_points[..., 2].masked_fill(~valid, float('inf'))

    # A reduction, not a scatter: the per-pixel minimum is what a point writes,
    # so no ordering of the point axis can change the outcome.
    winning_depth = torch.full(
        rendering_points.shape[:-2] + (render_height * render_width,),
        float('inf'),
        dtype=rendering_points.dtype,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=-1,
        index=pixel_index,
        src=depth_key,
        reduce='amin',
        include_self=True,
    )

    # Reduce the point indices the same way, so two points tying on depth resolve
    # to the lower index rather than to whichever one happened to be written last.
    point_index = torch.arange(
        num_points, dtype=torch.int64, device=rendering_points.device
    ).expand_as(pixel_index)
    owns_pixel = valid & (depth_key == winning_depth.gather(dim=-1, index=pixel_index))
    winner = torch.full(
        rendering_points.shape[:-2] + (render_height * render_width,),
        num_points,
        dtype=torch.int64,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=-1,
        index=pixel_index,
        src=torch.where(owns_pixel, point_index, num_points),
        reduce='amin',
        include_self=True,
    )
    winner = winner.masked_fill(winner == num_points, -1)

    return winner.reshape(rendering_points.shape[:-2] + (render_height, render_width))
