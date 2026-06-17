"""Shared continuous-heatmap palette helpers for atomic displays."""

import torch


def map_scalars_to_rgb(scalars: torch.Tensor) -> torch.Tensor:
    """Map non-negative scalars to RGB via a fixed continuous heatmap palette.

    Normalizes ``scalars`` by its maximum value (clamped at a tiny epsilon)
    so the palette spans the input range, then evaluates a perceptual
    blue->cyan->green->yellow->red ramp at every entry. Shape-agnostic:
    the output is ``scalars.shape + (3,)`` so per-vertex 1-D inputs
    produce ``(V, 3)`` and per-texel 2-D inputs produce ``(H, W, 3)``.

    Args:
        scalars: Tensor of non-negative scalar values. Any shape; dtype is
            cast internally to float64 for the normalization.

    Returns:
        RGB tensor of shape ``scalars.shape + (3,)``, uint8 dtype with
        channel values in ``[0, 255]``.
    """
    assert isinstance(scalars, torch.Tensor), (
        "Scalars must be a torch.Tensor. scalars=%r" % scalars
    )
    assert scalars.numel() > 0, "Scalars tensor must be non-empty. scalars.shape=%r" % (
        scalars.shape,
    )
    assert bool((scalars >= 0).all()), (
        "Scalars must be non-negative. scalars.min()=%r" % scalars.min().item()
    )

    flat = scalars.detach().to(torch.float64).reshape(-1)
    max_value = float(flat.max().item())
    normalized = flat / max(max_value, 1e-12)

    stops = torch.tensor(
        [0.0, 0.25, 0.50, 0.75, 1.0],
        dtype=torch.float64,
    )
    palette = torch.tensor(
        [
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ],
        dtype=torch.float64,
    )

    segment_index = torch.bucketize(
        normalized.clamp(min=0.0, max=1.0),
        stops[1:-1],
    )
    left_stop = stops[segment_index]
    right_stop = stops[segment_index + 1]
    segment_extent = (right_stop - left_stop).clamp(min=1e-12)
    segment_fraction = ((normalized - left_stop) / segment_extent).clamp(
        min=0.0, max=1.0
    )
    left_color = palette[segment_index]
    right_color = palette[segment_index + 1]
    interpolated = left_color + (right_color - left_color) * segment_fraction.unsqueeze(
        -1
    )

    rgb_uint8 = interpolated.round().clamp(min=0.0, max=255.0).to(torch.uint8)
    return rgb_uint8.reshape(*scalars.shape, 3)
