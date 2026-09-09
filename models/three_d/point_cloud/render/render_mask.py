"""Mask rendering from point clouds using projection methods."""

from typing import Optional, Tuple

import torch

from models.three_d.point_cloud.render.common.select_nearest_point_per_pixel import (
    select_nearest_point_per_pixel,
)


def render_mask_from_rendering_points(
    rendering_points: torch.Tensor,
    resolution: Tuple[int, int],
    device: torch.device,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mark the pixels a surviving point landed on.

    Args:
        rendering_points: Pre-processed points [..., N, 3] float torch.Tensor of
            (x, y, depth), the point axis in pc.xyz order and the leading axes
            enumerating the cameras rendered.
        resolution: Target resolution as (height, width) tuple.
        device: Device for the tensor.
        valid: Optional [..., N] bool torch.Tensor marking which points each
            camera keeps; None means every point of rendering_points is marked.

    Returns:
        Boolean mask torch.Tensor of shape [..., H, W] indicating valid pixels,
        carrying the leading axes of rendering_points.
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

    winner = select_nearest_point_per_pixel(
        rendering_points=rendering_points,
        valid=valid,
        resolution=resolution,
    )

    return (winner >= 0).to(device)
