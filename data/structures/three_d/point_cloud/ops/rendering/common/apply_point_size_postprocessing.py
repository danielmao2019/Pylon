import torch
from typing import Union
from data.structures.three_d.point_cloud.ops.rendering.common.create_circular_kernel_offsets import (
    create_circular_kernel_offsets,
)


def apply_point_size_postprocessing(
    rendered_image: torch.Tensor,
    depth_map: torch.Tensor,
    point_size: float,
    ignore_value: Union[int, float] = 0.0,
) -> torch.Tensor:
    """Apply point size effect through depth-aware dilation post-processing.

    For each pixel, look in a circular neighborhood and propagate the value
    from the pixel with minimum (closest) depth.

    Args:
        rendered_image: Rendered image tensor [C, H, W] or [H, W]
        depth_map: Depth map tensor [H, W]
        point_size: Size of circular neighborhood
        ignore_value: Value representing no data/background

    Returns:
        Post-processed image tensor with same shape as input
    """
    if point_size <= 1.0:
        return rendered_image

    device = rendered_image.device
    is_multichannel = rendered_image.ndim == 3
    H, W = rendered_image.shape[-2:]

    result = rendered_image.clone()
    kernel_offsets = create_circular_kernel_offsets(point_size, device)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij'
    )

    for dy, dx in kernel_offsets:
        neighbor_y = y_coords + dy
        neighbor_x = x_coords + dx

        valid_mask = (
            (neighbor_y >= 0) & (neighbor_y < H) & (neighbor_x >= 0) & (neighbor_x < W)
        )

        if not valid_mask.any():
            continue

        curr_y = y_coords[valid_mask]
        curr_x = x_coords[valid_mask]
        neighbor_y = neighbor_y[valid_mask]  # Reuse neighbor_y
        neighbor_x = neighbor_x[valid_mask]  # Reuse neighbor_x

        neighbor_depths = depth_map[neighbor_y, neighbor_x]
        current_depths = depth_map[curr_y, curr_x]

        propagate_mask = (neighbor_depths != ignore_value) & (
            (current_depths == ignore_value) | (neighbor_depths < current_depths)
        )

        if propagate_mask.any():
            curr_y = curr_y[propagate_mask]  # Reuse curr_y for update_y
            curr_x = curr_x[propagate_mask]  # Reuse curr_x for update_x
            neighbor_y = neighbor_y[propagate_mask]  # Reuse neighbor_y for source_y
            neighbor_x = neighbor_x[propagate_mask]  # Reuse neighbor_x for source_x

            if is_multichannel:
                result[:, curr_y, curr_x] = rendered_image[:, neighbor_y, neighbor_x]
            else:
                result[curr_y, curr_x] = rendered_image[neighbor_y, neighbor_x]

    return result
