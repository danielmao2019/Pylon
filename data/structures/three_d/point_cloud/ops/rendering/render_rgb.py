"""RGB rendering from point clouds using projection methods."""

from typing import Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.ops.rendering.common.apply_point_size_postprocessing import (
    apply_point_size_postprocessing,
)
from data.structures.three_d.point_cloud.ops.rendering.common.prepare_points_for_rendering import (
    prepare_points_for_rendering,
)
from data.structures.three_d.point_cloud.ops.rendering.common.validate_rendering_inputs import (
    validate_rendering_inputs,
)
from data.structures.three_d.point_cloud.ops.rendering.render_depth import (
    render_depth_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_mask import (
    render_mask_from_rendering_points,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def render_rgb_from_rendering_points(
    rendering_points: torch.Tensor,
    original_data_indices: torch.Tensor,
    pc: PointCloud,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
) -> torch.Tensor:
    """Render RGB image from pre-processed rendering points.

    Args:
        rendering_points: Pre-processed points [M, 3] with (x, y, depth).
        original_data_indices: Indices mapping rendered points to original data [M].
        pc: Point cloud containing 'rgb' field with color information.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 0.0).

    Returns:
        RGB image tensor of shape [3, H, W] with normalized values in [0, 1].

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

    # Get colors for visible points
    pixel_colors = colors[original_data_indices]

    # Allocate RGB image
    rgb_image = torch.full(
        (3, render_height, render_width),
        ignore_value,
        dtype=torch.float32,
        device=rendering_points.device,
    )

    # Render pixels
    rgb_image[:, rendering_points[:, 1].long(), rendering_points[:, 0].long()] = (
        pixel_colors.T.float()
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

    Projects 3D point cloud coordinates with RGB colors onto 2D image plane
    using camera parameters and generates an RGB image. Supports circular
    point rendering for improved visualization.

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
        NotImplementedError: If convention other than "opengl" is specified.
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
    rendering_points, original_data_indices = prepare_points_for_rendering(
        pc=pc,
        camera=camera,
        resolution=resolution,
    )

    # Render RGB image
    rgb_image = render_rgb_from_rendering_points(
        rendering_points=rendering_points,
        original_data_indices=original_data_indices,
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
        )

        rgb_image = apply_point_size_postprocessing(
            rendered_image=rgb_image,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )

    # Handle mask creation if requested
    if return_mask:
        valid_mask = render_mask_from_rendering_points(
            rendering_points=rendering_points,
            resolution=resolution,
            device=rendering_points.device,
        )

        # Apply point size processing to mask if needed
        if point_size > 1.0:
            depth_map = render_depth_from_rendering_points(
                rendering_points=rendering_points,
                resolution=resolution,
                ignore_value=float('inf'),
                return_mask=False,
            )

            valid_mask = apply_point_size_postprocessing(
                rendered_image=valid_mask.float(),
                depth_map=depth_map,
                point_size=point_size,
                ignore_value=float('inf'),
            ).bool()

        return rgb_image, valid_mask
    else:
        return rgb_image
