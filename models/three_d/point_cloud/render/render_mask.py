"""Mask rendering from point clouds using projection methods."""

from typing import Optional, Tuple

import torch


def render_mask_from_rendering_points(
    rendering_points: torch.Tensor,
    resolution: Tuple[int, int],
    device: torch.device,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create a valid pixel mask from rendered points.

    Args:
        rendering_points: Pre-processed points [..., N, 3] float torch.Tensor of
            (x, y, depth), the leading axes enumerating the cameras rendered.
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

    render_height, render_width = resolution

    # Allocate mask, its leading axes those of rendering_points
    valid_mask = torch.zeros(
        rendering_points.shape[:-2] + (render_height, render_width),
        dtype=torch.bool,
        device=device,
    )

    # Mark valid pixels, each index selected by valid so a culled point marks nothing
    selector = torch.nonzero(valid, as_tuple=True)
    valid_mask[
        selector[:-1]
        + (
            rendering_points[..., 1][selector].long(),
            rendering_points[..., 0][selector].long(),
        )
    ] = True

    return valid_mask
