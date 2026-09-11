"""Normal map rendering from point clouds using projection methods."""

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
    render_depth_from_point_cloud,
    render_depth_from_rendering_points,
)
from models.three_d.point_cloud.render.render_mask import (
    render_mask_from_rendering_points,
)
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

    First renders depth map from point cloud, then computes normals from depth gradients. Output normals are in OpenCV camera coordinate system.

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
    intrinsics = camera.intrinsics
    intrinsics_matrix = torch.tensor(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=intrinsics.device,
    )
    return depth_to_normals(
        depth_map=depth_map,
        camera_intrinsics=intrinsics_matrix,
        depth_ignore_value=float('inf'),
        normal_ignore_value=ignore_value,
        return_mask=return_mask,
    )


def render_normal_from_rendering_points_3d(
    rendering_points: torch.Tensor,
    valid: torch.Tensor,
    pc_data: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
    ignore_value: float = 0.0,
) -> torch.Tensor:
    """Render normal map from pre-processed rendering points using 3D approach.

    Args:
        rendering_points: Pre-processed points [N, 3] float torch.Tensor of (x, y, depth), the point axis in pc_data.xyz order.
        valid: [N] bool torch.Tensor marking which points the camera keeps; a point marked False never owns a pixel.
        pc_data: Point cloud containing a 'normals' field of world-space normals.
        camera: Camera containing extrinsics and convention; the rendered normals come back in its OpenCV camera frame.
        resolution: Target resolution as (height, width) tuple.
        ignore_value: Fill value for pixels with no projections (default: 0.0).

    Returns:
        Normal map torch.Tensor of shape [3, H, W], float32, with unit-norm normal vectors.

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

    # Read the normal of the point that owns each pixel
    visible_world_normals = world_normals[nearest_point_index.clamp(min=0)]  # [H, W, 3]

    # Transform normals from world to camera coordinates

    # Convert camera extrinsics to OpenCV convention
    camera = camera.to(device=rendering_points.device, extr_convention="opencv")
    rotation_matrix = camera.extrinsics.w2c[:3, :3]

    # Transform normals to camera coordinates (rotation only)
    camera_normals = torch.matmul(visible_world_normals, rotation_matrix.T)

    # Normalize after transformation
    camera_normals = torch.nn.functional.normalize(camera_normals, dim=-1)

    # Move the channel axis in front of the image axes and blank the unowned pixels
    normal_map = torch.where(
        (nearest_point_index >= 0).unsqueeze(-3),
        camera_normals.movedim(-1, -3).float(),
        torch.tensor(ignore_value, dtype=torch.float32, device=rendering_points.device),
    )

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

    Assumes pc contains 'normals' key with pre-computed normals in world coordinates. Transforms normals to OpenCV camera coordinate system and renders them.

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
    rendering_points, valid = prepare_points_for_rendering(
        pc=pc,
        camera=camera,
        resolution=resolution,
    )

    # Render normal map
    normal_map = render_normal_from_rendering_points_3d(
        rendering_points=rendering_points,
        valid=valid,
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

        normal_map = apply_point_size_postprocessing(
            rendered_image=normal_map,
            depth_map=depth_map,
            point_size=point_size,
            ignore_value=float('inf'),
        )

        # The helper fills with the depth sentinel it was handed, which is not this renderer's own background
        normal_map = normal_map.masked_fill(~covered.unsqueeze(-3), ignore_value)

        # Re-normalize after dilation, only at the pixels carrying a normal rather than this renderer's background
        valid_pixels = (normal_map != ignore_value).any(dim=-3)
        normal_map[:, valid_pixels] = torch.nn.functional.normalize(
            normal_map[:, valid_pixels], dim=0
        )

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

        return normal_map, valid_mask
    else:
        return normal_map
