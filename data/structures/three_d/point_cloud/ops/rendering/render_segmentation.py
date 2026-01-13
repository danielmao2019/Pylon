"""Segmentation rendering from point clouds using projection methods."""

from typing import Tuple, Union

import torch
from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.ops.rendering.common import (
    apply_point_size_postprocessing,
    prepare_points_for_rendering,
    validate_rendering_inputs,
)
from data.structures.three_d.point_cloud.ops.rendering.render_depth import (
    render_depth_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_mask import (
    render_mask_from_rendering_points,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def render_segmentation_from_rendering_points(
    rendering_points: torch.Tensor,
    original_data_indices: torch.Tensor,
    pc: PointCloud,
    key: str,
    resolution: Tuple[int, int],
    ignore_value: int = 255,
) -> torch.Tensor:
    """Render segmentation map from pre-processed rendering points.

    Args:
        rendering_points: Pre-processed points [M, 3] with (x, y, depth).
        original_data_indices: Indices mapping rendered points to original data [M].
        pc: Point cloud containing segmentation labels under specified key.
        key: Key name for segmentation labels in pc.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 255).

    Returns:
        Segmentation map tensor of shape [H, W] with integer labels.

    Raises:
        AssertionError: If labels tensor is empty.
    """
    assert hasattr(pc, key), f"PointCloud missing '{key}' field"
    render_height, render_width = resolution
    labels = getattr(pc, key)
    assert (
        labels.numel() > 0
    ), f"Labels tensor must not be empty, got {labels.numel()} elements"

    # Get labels for visible points
    pixel_labels = labels[original_data_indices]

    # Allocate segmentation map
    seg_map = torch.full(
        (render_height, render_width),
        ignore_value,
        dtype=torch.int64,
        device=rendering_points.device,
    )

    # Render pixels
    seg_map[rendering_points[:, 1].long(), rendering_points[:, 0].long()] = (
        pixel_labels.to(torch.int64)
    )

    return seg_map


def render_segmentation_from_point_cloud(
    pc: PointCloud,
    key: str,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: int = 255,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render segmentation map from point cloud using camera projection.

    Projects 3D point cloud coordinates with segmentation labels onto 2D image
    plane using camera parameters. Creates a pixel-wise segmentation map with
    support for circular point rendering for improved visualization.

    Args:
        pc_data: Point cloud data containing xyz coordinates and segmentation labels.
        key: Key name for segmentation labels in pc_data.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 255).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            Segmentation map tensor of shape [H, W] with integer labels.
        If return_mask is True:
            Tuple of (segmentation map tensor, valid mask tensor of shape [H, W]).

    Raises:
        AssertionError: If point cloud is empty, labels are missing, or no points project within bounds.
        NotImplementedError: If convention other than "opengl" is specified.
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert hasattr(pc, key), f"PointCloud must contain '{key}' field"

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

    # Render segmentation map
    seg_map = render_segmentation_from_rendering_points(
        rendering_points=rendering_points,
        original_data_indices=original_data_indices,
        pc=pc,
        key=key,
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

        seg_map = apply_point_size_postprocessing(
            rendered_image=seg_map.float(),
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        ).long()

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

        return seg_map, valid_mask
    else:
        return seg_map
