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

    # A background pixel carries positive infinity, so it is never a source nearer than a rendered one.
    source_depth = depth_map.masked_fill(depth_map == ignore_value, float('inf'))

    # The [..., num_offsets, H, W] stack of source_depth shifted by every kernel offset, read off a positive-infinity border as wide as the kernel's reach, so a shift leaving the image reads positive infinity.
    reach = int(kernel_offsets.abs().max())
    neighbor_depth = torch.nn.functional.pad(
        source_depth, pad=(reach, reach, reach, reach), value=float('inf')
    )[
        ...,
        kernel_offsets[:, 0].reshape(-1, 1, 1)
        + reach
        + torch.arange(render_height, device=rendered_image.device).reshape(-1, 1),
        kernel_offsets[:, 1].reshape(-1, 1, 1)
        + reach
        + torch.arange(render_width, device=rendered_image.device),
    ]

    # The offset axis' argmin names, for each pixel, which shifted source is nearest.
    source_offset = neighbor_depth.min(dim=-3).indices

    # Each pixel reads the value at its own flat index shifted by the offset source_offset names, one index broadcast across the channel axis when there is one; a pixel no disc reached may shift out of the image, so its index is clamped back in and its value blanked below.
    dilated_image = torch.take_along_dim(
        rendered_image.flatten(start_dim=-2),
        (
            torch.arange(render_height * render_width, device=rendered_image.device)
            + (kernel_offsets[:, 0] * render_width + kernel_offsets[:, 1])[
                source_offset
            ].flatten(start_dim=-2)
        )
        .clamp(min=0, max=render_height * render_width - 1)
        .reshape(source_offset.shape[:-2] + (1,) * channel_axis + (-1,)),
        dim=-1,
    ).reshape(rendered_image.shape)

    # A pixel whose nearest source is still positive infinity was reached by no disc, so it keeps the background.
    dilated_image = dilated_image.masked_fill(
        torch.isinf(neighbor_depth.amin(dim=-3)).reshape(
            depth_map.shape[:-2] + (1,) * channel_axis + depth_map.shape[-2:]
        ),
        ignore_value,
    )

    return dilated_image
