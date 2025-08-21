"""Depth rendering from point clouds using projection methods."""

import torch
from typing import Dict, Tuple


def render_depth_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_value: float = -1.0
) -> torch.Tensor:
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

    Returns:
        Depth map tensor of shape [H, W] with depth values in camera coordinate system

    Raises:
        AssertionError: If point cloud is empty or no points project within image bounds
        NotImplementedError: If convention other than "opengl" is specified
    """
    # Step 1: Input validation
    assert isinstance(pc_data, dict), f"pc_data must be dict, got {type(pc_data)}"
    assert 'pos' in pc_data, f"pc_data must contain 'pos' key, got keys: {list(pc_data.keys())}"

    points = pc_data['pos']

    # Validate points tensor
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"points must be 2D tensor with shape (N, 3), got shape {points.shape}"
    assert points.shape[1] == 3, f"points must have 3 coordinates (XYZ), got shape {points.shape}"
    assert points.shape[0] > 0, f"Point cloud cannot be empty, got {points.shape[0]} points"

    # Validate camera matrices
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (3, 3), f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert camera_extrinsics.shape == (4, 4), f"camera_extrinsics must be 4x4 matrix, got shape {camera_extrinsics.shape}"

    # Validate all tensors are on same device
    assert points.device == camera_intrinsics.device, f"points device {points.device} != camera_intrinsics device {camera_intrinsics.device}"
    assert points.device == camera_extrinsics.device, f"points device {points.device} != camera_extrinsics device {camera_extrinsics.device}"

    # Validate resolution
    assert isinstance(resolution, (tuple, list)), f"resolution must be tuple or list, got {type(resolution)}"
    assert len(resolution) == 2, f"resolution must have 2 elements (width, height), got {len(resolution)}"
    assert all(isinstance(x, int) and x > 0 for x in resolution), f"resolution must be positive integers, got {resolution}"

    # Validate convention
    assert isinstance(convention, str), f"convention must be str, got {type(convention)}"
    assert convention in ["opengl", "standard"], f"convention must be 'opengl' or 'standard', got '{convention}'"

    # Validate ignore_value
    assert isinstance(ignore_value, (int, float)), f"ignore_value must be int or float, got {type(ignore_value)}"

    render_width, render_height = resolution

    # Step 2: Device and dtype conversions
    points = points.clone().to(dtype=torch.float64)
    camera_intrinsics = camera_intrinsics.clone().to(device=points.device, dtype=torch.float64)
    camera_extrinsics = camera_extrinsics.clone().to(device=points.device, dtype=torch.float64)

    # Step 3: Scale camera intrinsics in-place
    original_width = int(camera_intrinsics[0, 2] * 2)   # Estimate from principal point cx
    original_height = int(camera_intrinsics[1, 2] * 2)  # Estimate from principal point cy
    scale_x = render_width / original_width
    scale_y = render_height / original_height

    camera_intrinsics[0, 0] *= scale_x  # Scale focal length fx
    camera_intrinsics[1, 1] *= scale_y  # Scale focal length fy
    camera_intrinsics[0, 2] *= scale_x  # Scale principal point cx
    camera_intrinsics[1, 2] *= scale_y  # Scale principal point cy

    # Assert last row is [0, 0, 1] to ensure depth preservation during projection
    assert torch.allclose(
        camera_intrinsics[2, :],
        torch.tensor([0.0, 0.0, 1.0], device=camera_intrinsics.device, dtype=camera_intrinsics.dtype)
    ), f"Camera intrinsics last row must be [0, 0, 1], got {camera_intrinsics[2, :]}"

    # Step 4: Camera convention conversion
    if convention == "opengl":
        # Convert to world-to-camera by inverting camera-to-world extrinsics
        # Use materialize_tensor to avoid lazy wrapper issues
        from utils.ops.materialize_tensor import materialize_tensor
        materialized_extrinsics = materialize_tensor(camera_extrinsics)
        world_to_camera = torch.inverse(materialized_extrinsics)
    else:
        raise NotImplementedError("Standard convention not implemented yet")

    # Step 5: Transform points into camera local frame
    # Use addmm for memory efficiency: points @ R.T + t
    points = torch.addmm(world_to_camera[:3, 3], points, world_to_camera[:3, :3].T)

    # Step 6: Filter points based on z coordinate (OpenGL: negative Z in front)
    points = points[points[:, 2] < 0]
    assert len(points) > 0, f"No points in front of camera for depth rendering, got {len(points)} valid points"

    # Step 7: Project points into 2D
    # Matrix multiplication preserves depth column due to [0, 0, 1] row assertion above
    points = (camera_intrinsics @ points.T).T

    # Perspective division and horizontal flip
    points[:, 0] /= points[:, 2]  # Perspective division for x
    points[:, 1] /= points[:, 2]  # Perspective division for y
    points[:, 0] = (render_width - 1) - points[:, 0]  # Apply horizontal flip correction

    # Step 8: Filter points within image bounds
    points = points[
        (points[:, 0] >= 0) & (points[:, 0] < render_width) &
        (points[:, 1] >= 0) & (points[:, 1] < render_height)
    ]
    assert len(points) > 0, f"No points within image bounds for depth rendering, got {len(points)} valid points"

    # Step 9: Sort points by depth (closest first)
    points = points[torch.argsort(torch.abs(points[:, 2]))]

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
    return depth_map.to(dtype=torch.float32)
