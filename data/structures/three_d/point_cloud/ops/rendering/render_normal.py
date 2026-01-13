"""Normal map rendering from point clouds using projection methods."""

from typing import Tuple, Union

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.ops.rendering.common import (
    apply_point_size_postprocessing,
    prepare_points_for_rendering,
    validate_rendering_inputs,
)
from data.structures.three_d.point_cloud.ops.rendering.render_depth import (
    render_depth_from_point_cloud,
    render_depth_from_rendering_points,
)
from data.structures.three_d.point_cloud.ops.rendering.render_mask import (
    render_mask_from_rendering_points,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from utils.conversions.depth_to_normals import depth_to_normals


def render_normal_from_point_cloud_2d(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render normal map from point cloud using 2D depth-based approach.

    First renders depth map from point cloud, then computes normals from depth gradients.
    Output normals are in OpenCV camera coordinate system.

    Args:
        pc: Point cloud containing 3D coordinates.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no projections (default: 0.0).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            Normal map tensor of shape [3, H, W] with normalized normal vectors.
        If return_mask is True:
            Tuple of (normal map [3, H, W], valid mask [H, W]).
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"

    # Render depth map
    depth_map = render_depth_from_point_cloud(
        pc=pc,
        camera=camera,
        resolution=resolution,
        ignore_value=float('inf'),
        return_mask=False,
        point_size=point_size,
    )

    # Convert depth to normals
    return depth_to_normals(
        depth_map=depth_map,
        camera_intrinsics=camera.intrinsics,
        depth_ignore_value=float('inf'),
        normal_ignore_value=ignore_value,
        return_mask=return_mask,
    )


def render_normal_from_rendering_points_3d(
    rendering_points: torch.Tensor,
    original_data_indices: torch.Tensor,
    pc_data: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
) -> torch.Tensor:
    """Render normal map from pre-processed rendering points using 3D approach.

    Args:
        rendering_points: Pre-processed points [M, 3] with (x, y, depth).
        original_data_indices: Indices mapping rendered points to original data [M].
        pc_data: Point cloud dictionary containing 'normals' key.
        camera: Camera containing extrinsics and convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no projections (default: 0.0).

    Returns:
        Normal map tensor of shape [3, H, W] with normalized normal vectors.

    Raises:
        AssertionError: If normals count doesn't match points count or normals aren't 3D.
    """
    render_height, render_width = resolution
    world_normals = pc_data.normals
    assert (
        world_normals.shape[0] == pc_data.xyz.shape[0]
    ), f"Normals count {world_normals.shape[0]} must match points count {pc_data.xyz.shape[0]}"
    assert (
        world_normals.shape[1] == 3
    ), f"Normals must be 3D vectors, got shape {world_normals.shape}"

    # Normalize world normals
    world_normals = torch.nn.functional.normalize(world_normals, dim=-1)

    # Get normals for visible points
    visible_world_normals = world_normals[original_data_indices]

    # Transform normals from world to camera coordinates

    # Convert camera extrinsics to OpenCV convention
    camera = camera.to(device=rendering_points.device, convention="opencv")
    rotation_matrix = camera.w2c[:3, :3]

    # Transform normals to camera coordinates (rotation only)
    camera_normals = torch.matmul(visible_world_normals, rotation_matrix.T)

    # Normalize after transformation
    camera_normals = torch.nn.functional.normalize(camera_normals, dim=-1)

    # Allocate normal map
    normal_map = torch.full(
        (3, render_height, render_width),
        ignore_value,
        dtype=torch.float32,
        device=rendering_points.device,
    )

    # Render pixels
    pixel_coords_y = rendering_points[:, 1].long()
    pixel_coords_x = rendering_points[:, 0].long()
    normal_map[:, pixel_coords_y, pixel_coords_x] = camera_normals.float().T

    return normal_map


def render_normal_from_point_cloud_3d(
    pc: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
    return_mask: bool = False,
    point_size: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render normal map from point cloud using 3D approach.

    Assumes pc contains 'normals' key with pre-computed normals in world coordinates.
    Transforms normals to OpenCV camera coordinate system and renders them.

    Args:
        pc: Point cloud containing 'normals' field.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no projections (default: 0.0).
        return_mask: If True, also return valid pixel mask (default: False).
        point_size: Size of rendered points in pixels (default: 1.0).

    Returns:
        If return_mask is False:
            Normal map tensor of shape [3, H, W] with normalized normal vectors.
        If return_mask is True:
            Tuple of (normal map [3, H, W], valid mask [H, W]).

    Raises:
        AssertionError: If normals are missing or dimensions don't match.
    """
    assert isinstance(pc, PointCloud), f"{type(pc)=}"
    assert hasattr(pc, 'normals'), "PointCloud must contain normals field"

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

    # Render normal map
    normal_map = render_normal_from_rendering_points_3d(
        rendering_points=rendering_points,
        original_data_indices=original_data_indices,
        pc_data=pc,
        camera=camera,
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

        normal_map = apply_point_size_postprocessing(
            rendered_image=normal_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )

        # Re-normalize after dilation
        valid_pixels = (normal_map != ignore_value).any(dim=0)
        normal_map[:, valid_pixels] = torch.nn.functional.normalize(
            normal_map[:, valid_pixels], dim=0
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

        return normal_map, valid_mask
    else:
        return normal_map
