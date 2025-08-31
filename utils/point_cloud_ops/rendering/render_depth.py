"""Depth rendering from point clouds using projection methods."""

import torch
from typing import Dict, Tuple, Union
from .render_common import prepare_points_for_rendering


def render_depth_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_value: float = -1.0,
    return_mask: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render depth map from point cloud using camera projection.

    Projects 3D point cloud coordinates onto 2D image plane using camera
    parameters and generates a depth map. Supports OpenGL camera convention
    with automatic intrinsics scaling for target resolution.

    Args:
        pc_data: Point cloud dictionary containing 'pos' key with 3D coordinates
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world transform)
        resolution: Target resolution as (width, height) tuple - intrinsics scaled automatically
        convention: Camera extrinsics convention ("opengl" supported, "standard" not implemented)
        ignore_value: Fill value for pixels with no point projections (default: -1.0)
        return_mask: If True, also return valid pixel mask (default: False)

    Returns:
        If return_mask is False:
            Depth map tensor of shape [H, W] with depth values in camera coordinate system
        If return_mask is True:
            Tuple of (depth map tensor, valid mask tensor of shape [H, W] with boolean values)

    Raises:
        AssertionError: If point cloud is empty or no points project within image bounds
        NotImplementedError: If convention other than "opengl" is specified
    """
    # Depth-specific validation
    assert isinstance(ignore_value, (int, float)), f"ignore_value must be int or float, got {type(ignore_value)}"
    assert isinstance(return_mask, bool), f"return_mask must be bool, got {type(return_mask)}"

    render_width, render_height = resolution

    # Use common preprocessing
    points, filtered_indices = prepare_points_for_rendering(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention=convention
    )

    # Step 10: Allocate depth map
    depth_map = torch.full(
        (render_height, render_width),
        ignore_value,
        dtype=torch.float64,
        device=points.device
    )

    # Step 11: Fill depth map with sorted depths (use absolute values for positive depths)
    depth_map[points[:, 1].long(), points[:, 0].long()] = torch.abs(points[:, 2])

    # Convert to float32 for final output
    depth_map = depth_map.to(dtype=torch.float32)

    if return_mask:
        # Step 12: Create valid mask
        valid_mask = torch.zeros(
            (render_height, render_width),
            dtype=torch.bool,
            device=points.device
        )
        valid_mask[points[:, 1].long(), points[:, 0].long()] = True
        
        return depth_map, valid_mask
    else:
        return depth_map
