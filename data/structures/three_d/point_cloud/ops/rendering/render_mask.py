"""Mask rendering from point clouds using projection methods."""

import torch
from typing import Tuple


def render_mask_from_rendering_points(
    rendering_points: torch.Tensor, resolution: Tuple[int, int], device: torch.device
) -> torch.Tensor:
    """Create a valid pixel mask from rendered points.

    Args:
        rendering_points: Pre-processed points [M, 3] with (x, y, depth).
        resolution: Target resolution as (height, width) tuple.
        device: Device for the tensor.

    Returns:
        Boolean mask tensor of shape [H, W] indicating valid pixels.
    """
    render_height, render_width = resolution

    # Allocate mask
    valid_mask = torch.zeros(
        (render_height, render_width), dtype=torch.bool, device=device
    )

    # Mark valid pixels
    valid_mask[rendering_points[:, 1].long(), rendering_points[:, 0].long()] = True

    return valid_mask
