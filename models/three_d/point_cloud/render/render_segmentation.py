"""Segmentation rendering from point clouds using projection methods."""

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
from models.three_d.point_cloud.render.common.select_nearest_point_per_pixel import (
    select_nearest_point_per_pixel,
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


def render_segmentation_from_rendering_points(
    rendering_points: torch.Tensor,
    valid: torch.Tensor,
    pc: PointCloud,
    key: str,
    resolution: Tuple[int, int],
    ignore_value: int = 255,
) -> torch.Tensor:
    """Render segmentation map from pre-processed rendering points.

    Args:
        rendering_points: Pre-processed points [..., N, 3] float torch.Tensor of
            (x, y, depth), the point axis in pc.xyz order and the leading axes
            enumerating the cameras rendered.
        valid: [..., N] bool torch.Tensor marking which points each camera keeps.
        pc: Point cloud containing segmentation labels under specified key.
        key: Key name for segmentation labels in pc.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 255).

    Returns:
        Segmentation map torch.Tensor of shape [..., H, W], int64, carrying the
        leading axes of rendering_points.

    Raises:
        AssertionError: If labels tensor is empty.
    """
    assert hasattr(pc, key), f"PointCloud missing '{key}' field"
    labels = getattr(pc, key)
    assert (
        labels.numel() > 0
    ), f"Labels tensor must not be empty, got {labels.numel()} elements"

    # Resolve which point owns each pixel, then read that point's own label
    winner = select_nearest_point_per_pixel(
        rendering_points=rendering_points,
        valid=valid,
        resolution=resolution,
    )
    pixel_labels = labels[winner.clamp(min=0)]

    # Blank the pixels no surviving point landed on
    seg_map = torch.where(
        winner >= 0,
        pixel_labels.to(torch.int64),
        torch.tensor(ignore_value, dtype=torch.int64, device=rendering_points.device),
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
    rendering_points, valid = prepare_points_for_rendering(
        pc=pc,
        camera=camera,
        resolution=resolution,
    )

    # Render segmentation map
    seg_map = render_segmentation_from_rendering_points(
        rendering_points=rendering_points,
        valid=valid,
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
            valid=valid,
        )

        # The discs the dilation reaches are exactly the pixels the dilated depth
        # map keeps finite
        dilated_depth = apply_point_size_postprocessing(
            rendered_image=depth_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )
        covered = torch.isfinite(dilated_depth)

        seg_map = apply_point_size_postprocessing(
            rendered_image=seg_map.float(),
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )

        # The dilation leaves the depth sentinel outside those discs, and casting
        # it back to labels saturates in opposite directions per device, so this
        # renderer's own background goes back before the cast
        seg_map = seg_map.masked_fill(~covered, ignore_value).long()

    # Handle mask creation if requested
    if return_mask:
        if point_size > 1.0:
            # The dilation repainted the map, so the mask follows the discs it reached
            valid_mask = covered
        else:
            valid_mask = render_mask_from_rendering_points(
                rendering_points=rendering_points,
                resolution=resolution,
                device=rendering_points.device,
                valid=valid,
            )

        return seg_map, valid_mask
    else:
        return seg_map
