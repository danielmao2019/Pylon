"""RGB rendering from point clouds using projection methods."""

from typing import Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from models.three_d.point_cloud.render.common.apply_point_size_postprocessing import (
    apply_point_size_postprocessing,
)
from models.three_d.point_cloud.render.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from models.three_d.point_cloud.render.common.validate_rendering_inputs import (
    validate_rendering_inputs,
)
from models.three_d.point_cloud.render.render_depth import (
    render_depth_from_rendering_points,
)
from models.three_d.point_cloud.render.render_mask import (
    render_mask_from_rendering_points,
)


def render_rgb_from_rendering_points(
    rendering_points: torch.Tensor,
    valid: torch.Tensor,
    pc: PointCloud,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
) -> torch.Tensor:
    """Render RGB image from pre-processed rendering points.

    Args:
        rendering_points: Pre-processed points [N, 3] float torch.Tensor of (x, y, depth), the point axis in pc.xyz order.
        valid: [N] bool torch.Tensor marking which points the camera keeps; a point marked False never owns a pixel.
        pc: Point cloud containing 'rgb' field with color information.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 0.0).

    Returns:
        RGB image torch.Tensor of shape [3, H, W], float32, with normalized values in [0, 1].

    Raises:
        AssertionError: If colors tensor is empty.
    """
    assert hasattr(pc, 'rgb'), "PointCloud missing rgb field"
    render_height, render_width = resolution
    colors = pc.rgb
    assert (
        colors.numel() > 0
    ), f"Colors tensor must not be empty, got {colors.numel()} elements"

    # Normalize colors to [0, 1] range (keep float32 to reduce memory)
    colors = colors.clone()
    integer_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    is_integer_dtype = colors.dtype in integer_dtypes
    is_in_255_range = colors.min() >= 0 and colors.max() <= 255 and colors.max() > 1.0

    if is_integer_dtype or is_in_255_range:
        colors = colors / 255.0

    colors = torch.clamp(colors, 0.0, 1.0)

    # Resolve, per pixel, the valid point with the smallest depth landing there, reduced per pixel rather than scattered so occlusion does not depend on which write lands last. A culled point is parked on pixel 0, whose out-of-image coordinates are not scatterable, and its depth of positive infinity keeps it from ever owning that pixel.
    num_points = rendering_points.shape[-2]
    pixel_index = (
        rendering_points[..., 1].long() * render_width + rendering_points[..., 0].long()
    )
    pixel_index = pixel_index.masked_fill(~valid, 0)
    depth_key = rendering_points[..., 2].masked_fill(~valid, float('inf'))
    nearest_depth = torch.full(
        size=rendering_points.shape[:-2] + (render_height * render_width,),
        fill_value=float('inf'),
        dtype=rendering_points.dtype,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=-1,
        index=pixel_index,
        src=depth_key,
        reduce='amin',
        include_self=True,
    )
    # The point indices reduce the same way, so two points tying on depth resolve to the lower index.
    point_index = torch.arange(
        num_points, dtype=torch.int64, device=rendering_points.device
    ).expand_as(pixel_index)
    owns_pixel = valid & (depth_key == nearest_depth.gather(dim=-1, index=pixel_index))
    nearest_point_index = torch.full(
        size=rendering_points.shape[:-2] + (render_height * render_width,),
        fill_value=num_points,
        dtype=torch.int64,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=-1,
        index=pixel_index,
        src=torch.where(owns_pixel, point_index, num_points),
        reduce='amin',
        include_self=True,
    )
    nearest_point_index = nearest_point_index.masked_fill(
        nearest_point_index == num_points, -1
    ).reshape(rendering_points.shape[:-2] + (render_height, render_width))

    # Read the color of the point that owns each pixel, the color axis moved in front of the image axes
    pixel_colors = colors[nearest_point_index.clamp(min=0)].movedim(-1, -3)  # [3, H, W]

    # Blank the unowned pixels
    rgb_image = torch.where(
        (nearest_point_index >= 0).unsqueeze(-3),
        pixel_colors.float(),
        torch.tensor(ignore_value, dtype=torch.float32, device=rendering_points.device),
    )

    return rgb_image


def render_rgb_from_point_cloud(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render RGB image from point cloud using camera projection.

    Projects 3D point cloud coordinates with RGB colors onto 2D image plane using camera parameters and generates an RGB image. Supports circular point rendering for improved visualization.

    Args:
        pc: Point cloud data containing xyz and rgb fields.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 0.0).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            RGB image tensor of shape [3, H, W] with normalized values in [0, 1].
        If return_mask is True:
            Tuple of (RGB image tensor, valid mask tensor of shape [H, W]).

    Raises:
        AssertionError: If point cloud is empty, RGB data is missing, or no points project within bounds.
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert hasattr(pc, 'rgb'), "PointCloud must contain rgb field"

    # Validate inputs
    validate_rendering_inputs(
        pc=pc,
        camera=camera,
        resolution=resolution,
        ignore_value=ignore_value,
        return_mask=return_mask,
        point_size=point_size,
    )

    # Prepare points for rendering
    rendering_points, valid = prepare_points_for_rendering(
        pc=pc,
        camera=camera,
        resolution=resolution,
    )

    # Render RGB image
    rgb_image = render_rgb_from_rendering_points(
        rendering_points=rendering_points,
        valid=valid,
        pc=pc,
        resolution=resolution,
        ignore_value=ignore_value,
    )

    # Apply point size post-processing if needed
    if point_size > 1.0:
        depth_map = render_depth_from_rendering_points(
            rendering_points=rendering_points,
            resolution=resolution,
            ignore_value=float('inf'),
            return_mask=False,
            valid=valid,
        )

        # The discs the dilation reaches are exactly the pixels the dilated depth map keeps finite
        dilated_depth = apply_point_size_postprocessing(
            rendered_image=depth_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )
        covered = torch.isfinite(dilated_depth)

        rgb_image = apply_point_size_postprocessing(
            rendered_image=rgb_image,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )

        # The helper fills with the depth sentinel it was handed, which is not this renderer's own background
        rgb_image = rgb_image.masked_fill(~covered.unsqueeze(-3), ignore_value)

    # Handle mask creation if requested
    if return_mask:
        if point_size > 1.0:
            # The dilation repainted the image, so the mask follows the discs it reached
            valid_mask = covered
        else:
            valid_mask = render_mask_from_rendering_points(
                rendering_points=rendering_points,
                resolution=resolution,
                device=rendering_points.device,
                valid=valid,
            )

        return rgb_image, valid_mask
    else:
        return rgb_image
