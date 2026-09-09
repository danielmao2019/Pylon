import math

import torch


def create_circular_kernel_offsets(
    point_size: float, device: torch.device
) -> torch.Tensor:
    """Create offset positions for circular kernel.

    Args:
        point_size: Diameter of the circular kernel in pixels
        device: Device for the tensor

    Returns:
        Tensor of shape [num_pixels_in_circle, 2] with (y, x) offsets
    """
    kernel_radius = point_size / 2.0

    # The grid spans the same reach on both sides of the origin, so the disc it
    # carves is centred on the point rather than lopsided towards one corner.
    axis_offsets = torch.arange(
        -math.ceil(kernel_radius), math.ceil(kernel_radius) + 1, device=device
    )
    y_kernel, x_kernel = torch.meshgrid(axis_offsets, axis_offsets, indexing='ij')

    kernel_distances = torch.sqrt(x_kernel.float() ** 2 + y_kernel.float() ** 2)
    circular_mask = kernel_distances <= kernel_radius

    kernel_offsets = torch.stack(
        [y_kernel[circular_mask], x_kernel[circular_mask]], dim=1
    )

    return kernel_offsets
