"""Common utilities for point cloud rendering operations."""

import torch
from typing import Dict, Tuple


def prepare_points_for_rendering(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Common preprocessing for all point cloud rendering operations.
    
    Handles input validation, camera transformations, and point projection
    that are shared across RGB, depth, and segmentation rendering.
    
    Args:
        pc_data: Point cloud dictionary containing 'pos' key
        camera_intrinsics: 3x3 camera intrinsics matrix
        camera_extrinsics: 4x4 camera extrinsics matrix
        resolution: Target resolution as (width, height) tuple
        convention: Camera extrinsics convention ("opengl" supported)
    
    Returns:
        Tuple of (projected_points, sort_indices) where:
        - projected_points: 2D points [N, 3] sorted by depth (x, y, depth)
        - sort_indices: Original indices of points after filtering and sorting
    
    Raises:
        AssertionError: If validation fails or no points are visible
        NotImplementedError: If convention other than "opengl" is specified
    """
    render_width, render_height = resolution
    points = pc_data['pos']

    # Basic input validation (common to all)
    assert isinstance(pc_data, dict), f"pc_data must be dict, got {type(pc_data)}"
    assert 'pos' in pc_data, f"pc_data must contain 'pos' key, got keys: {list(pc_data.keys())}"
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"points must be 2D tensor with shape (N, 3), got shape {points.shape}"
    assert points.shape[1] == 3, f"points must have 3 coordinates (XYZ), got shape {points.shape}"
    assert points.shape[0] > 0, f"Point cloud cannot be empty, got {points.shape[0]} points"
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (3, 3), f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert camera_extrinsics.shape == (4, 4), f"camera_extrinsics must be 4x4 matrix, got shape {camera_extrinsics.shape}"
    assert points.device == camera_intrinsics.device, f"points device {points.device} != camera_intrinsics device {camera_intrinsics.device}"
    assert points.device == camera_extrinsics.device, f"points device {points.device} != camera_extrinsics device {camera_extrinsics.device}"
    assert isinstance(resolution, (tuple, list)), f"resolution must be tuple or list, got {type(resolution)}"
    assert len(resolution) == 2, f"resolution must have 2 elements (width, height), got {len(resolution)}"
    assert all(isinstance(x, int) and x > 0 for x in resolution), f"resolution must be positive integers, got {resolution}"
    assert isinstance(convention, str), f"convention must be str, got {type(convention)}"
    assert convention in ["opengl", "standard"], f"convention must be 'opengl' or 'standard', got '{convention}'"

    # Device and dtype conversions
    points = points.clone().to(dtype=torch.float64)
    camera_intrinsics = camera_intrinsics.clone().to(device=points.device, dtype=torch.float64)
    camera_extrinsics = camera_extrinsics.clone().to(device=points.device, dtype=torch.float64)

    # Scale camera intrinsics in-place
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

    # Camera convention conversion
    if convention == "opengl":
        # Convert to world-to-camera by inverting camera-to-world extrinsics
        from utils.ops.materialize_tensor import materialize_tensor
        materialized_extrinsics = materialize_tensor(camera_extrinsics)
        world_to_camera = torch.inverse(materialized_extrinsics)
    else:
        raise NotImplementedError("Standard convention not implemented yet")

    # Transform points into camera local frame
    points = torch.addmm(world_to_camera[:3, 3], points, world_to_camera[:3, :3].T)

    # Filter points based on z coordinate (OpenGL: negative Z in front)
    depth_mask = points[:, 2] < 0
    points = points[depth_mask]
    assert len(points) > 0, f"No points in front of camera for rendering, got {len(points)} valid points"

    # Project points into 2D
    points = (camera_intrinsics @ points.T).T

    # Perspective division and horizontal flip
    points[:, 0] /= points[:, 2]  # Perspective division for x
    points[:, 1] /= points[:, 2]  # Perspective division for y
    points[:, 0] = (render_width - 1) - points[:, 0]  # Apply horizontal flip correction

    # Filter points within image bounds
    bounds_mask = (
        (points[:, 0] >= 0) & (points[:, 0] < render_width) &
        (points[:, 1] >= 0) & (points[:, 1] < render_height)
    )
    points = points[bounds_mask]
    assert len(points) > 0, f"No points within image bounds for rendering, got {len(points)} valid points"

    # Sort points by depth (closest first)
    sort_indices = torch.argsort(torch.abs(points[:, 2]))
    points = points[sort_indices]
    
    # Create mapping from original indices through filtering to final sorted order
    original_indices = torch.arange(pc_data['pos'].shape[0], device=pc_data['pos'].device)
    filtered_indices = original_indices[depth_mask][bounds_mask][sort_indices]

    return points, filtered_indices