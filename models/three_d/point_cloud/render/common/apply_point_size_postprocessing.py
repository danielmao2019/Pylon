from typing import Union

import torch

from models.three_d.point_cloud.render.common.create_circular_kernel_offsets import (
    create_circular_kernel_offsets,
)


def apply_point_size_postprocessing(
    rendered_image: torch.Tensor,
    depth_map: torch.Tensor,
    point_size: float,
    ignore_value: Union[int, float],
) -> torch.Tensor:
    """Dilate each rendered point into a disc of point_size pixels.

    A nearer point's value overwrites a farther one, so the dilation respects the same occlusion the rasterizer resolved. The camera axes ride in front of the image axes, so one call dilates a whole batch and a single camera alike.

    Args:
        rendered_image: [..., C, H, W] or [..., H, W] float torch.Tensor of the rasterized values, its leading axes those of depth_map.
        depth_map: [..., H, W] float torch.Tensor of the depth the rasterizer resolved, ignore_value marking the pixels no point owns.
        point_size: Diameter of the circular kernel in pixels.
        ignore_value: Value marking no data, in both depth_map and the result.

    Returns:
        Dilated torch.Tensor of the same shape and dtype as rendered_image.
    """
    render_height, render_width = depth_map.shape[-2:]
    channel_axis = rendered_image.ndim == depth_map.ndim + 1

    kernel_offsets = create_circular_kernel_offsets(
        point_size=point_size, device=rendered_image.device
    )
    num_offsets = kernel_offsets.shape[0]

    # A background pixel carries positive infinity, so it is never a source nearer than a rendered one.
    source_depth = depth_map.masked_fill(depth_map == ignore_value, float('inf'))

    # Every pixel's disc of source pixels, out-of-image sources clamped back in and marked so their depth becomes positive infinity.
    y_coords, x_coords = torch.meshgrid(
        torch.arange(render_height, device=rendered_image.device),
        torch.arange(render_width, device=rendered_image.device),
        indexing='ij',
    )
    neighbor_y = y_coords + kernel_offsets[:, 0].reshape(num_offsets, 1, 1)
    neighbor_x = x_coords + kernel_offsets[:, 1].reshape(num_offsets, 1, 1)
    in_bounds = (
        (neighbor_y >= 0)
        & (neighbor_y < render_height)
        & (neighbor_x >= 0)
        & (neighbor_x < render_width)
    ).reshape(num_offsets, -1)
    source_index = (
        neighbor_y.clamp(min=0, max=render_height - 1) * render_width
        + neighbor_x.clamp(min=0, max=render_width - 1)
    ).reshape(num_offsets, -1)
    neighbor_depth = source_depth.reshape(source_depth.shape[:-2] + (-1,))[
        ..., source_index
    ].masked_fill(~in_bounds, float('inf'))

    # The offset axis' argmin names, for each pixel, which shifted source is nearest.
    nearest_depth, source_offset = neighbor_depth.min(dim=-2)
    source_flat = source_index[
        source_offset,
        torch.arange(render_height * render_width, device=rendered_image.device),
    ]

    image_flat = rendered_image.reshape(rendered_image.shape[:-2] + (-1,))
    if channel_axis:
        source_flat = source_flat.unsqueeze(-2).expand(image_flat.shape)
        nearest_depth = nearest_depth.unsqueeze(-2)
    dilated_image = torch.gather(image_flat, dim=-1, index=source_flat)
    dilated_image = dilated_image.masked_fill(
        torch.isinf(nearest_depth), ignore_value
    ).reshape(rendered_image.shape)

    return dilated_image
