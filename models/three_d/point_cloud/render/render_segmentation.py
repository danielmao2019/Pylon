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
from models.three_d.point_cloud.render.common.validate_rendering_inputs import (
    validate_rendering_inputs,
)
from models.three_d.point_cloud.render.render_depth import (
    render_depth_from_rendering_points,
)
from models.three_d.point_cloud.render.render_mask import (
    render_mask_from_rendering_points,
)


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

    Projects 3D point cloud coordinates with segmentation labels onto 2D image plane using camera parameters. Creates a pixel-wise segmentation map with support for circular point rendering for improved visualization.

    Args:
        pc: Point cloud data containing xyz coordinates and segmentation labels.
        key: Key name for segmentation labels in pc.
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

    # Prepare points for rendering; a single camera's validity is None, its culled points already dropped
    rendering_points, _, original_data_indices = prepare_points_for_rendering(
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

        # The discs the dilation reaches are exactly the pixels the dilated depth map keeps finite
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
        ).long()

        # The helper fills with the depth sentinel it was handed, which is not this renderer's own background
        seg_map = seg_map.masked_fill(~covered, ignore_value)

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
            )

        return seg_map, valid_mask
    else:
        return seg_map


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
        rendering_points: Pre-processed points [M, 3] float torch.Tensor of (x, y, depth), one row per point the camera kept.
        original_data_indices: [M] int64 torch.Tensor holding, for each row of rendering_points, the index of its point in pc.xyz.
        pc: Point cloud containing segmentation labels under specified key.
        key: Key name for segmentation labels in pc.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no point projections (default: 255).

    Returns:
        Segmentation map torch.Tensor of shape [H, W], int64.

    Raises:
        AssertionError: If labels tensor is empty.
    """
    assert hasattr(pc, key), f"PointCloud missing '{key}' field"
    render_height, render_width = resolution
    labels = getattr(pc, key)
    assert (
        labels.numel() > 0
    ), f"Labels tensor must not be empty, got {labels.numel()} elements"

    # One label per row of rendering_points
    labels = labels[original_data_indices]

    # Resolve, per pixel, the point with the smallest depth landing there, reduced per pixel rather than scattered so occlusion does not depend on which write lands last.
    num_points = rendering_points.shape[0]
    pixel_index = (
        rendering_points[:, 1].long() * render_width + rendering_points[:, 0].long()
    )
    nearest_depth = torch.full(
        size=(render_height * render_width,),
        fill_value=float('inf'),
        dtype=rendering_points.dtype,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=0,
        index=pixel_index,
        src=rendering_points[:, 2],
        reduce='amin',
        include_self=True,
    )
    # The point indices reduce the same way, so two points tying on depth resolve to the lower index.
    owning = torch.nonzero(
        rendering_points[:, 2] == nearest_depth[pixel_index], as_tuple=True
    )[0]
    nearest_point_index = torch.full(
        size=(render_height * render_width,),
        fill_value=num_points,
        dtype=torch.int64,
        device=rendering_points.device,
    ).scatter_reduce_(
        dim=0,
        index=pixel_index[owning],
        src=owning,
        reduce='amin',
        include_self=True,
    )
    nearest_point_index = nearest_point_index.masked_fill(
        nearest_point_index == num_points, -1
    ).reshape(render_height, render_width)

    # Read the label of the point that owns each pixel
    pixel_labels = labels[nearest_point_index.clamp(min=0)]

    # Blank the pixels no surviving point landed on
    seg_map = torch.where(
        nearest_point_index >= 0,
        pixel_labels.to(torch.int64),
        torch.tensor(ignore_value, dtype=torch.int64, device=rendering_points.device),
    )

    return seg_map
