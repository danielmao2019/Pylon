"""Segmentation rendering from point clouds using projection."""

import torch
from typing import Dict, Tuple


def render_segmentation_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_extrinsics: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_index: int = 255,
    key: str = "labels"
) -> torch.Tensor:
    """Render segmentation map from point cloud using camera projection.
    
    Projects 3D point cloud coordinates with segmentation labels onto 2D image
    plane using camera parameters. Creates a pixel-wise segmentation map where
    each pixel contains the label of the closest projected point.
    
    Args:
        pc_data: Point cloud dictionary containing 'pos' key with 3D coordinates
                 and segmentation labels under the specified key
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world transform)
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        resolution: Target resolution as (width, height) tuple - intrinsics scaled automatically
        convention: Camera extrinsics convention ("opengl" supported, "standard" not implemented)
        ignore_index: Fill value for pixels with no point projections (default: 255)
        key: Key name for segmentation labels in pc_data (default: "labels")
        
    Returns:
        Segmentation map tensor of shape [H, W] with integer labels
        
    Raises:
        AssertionError: If point cloud is empty, labels are missing, or no points project within bounds
        NotImplementedError: If convention other than "opengl" is specified
    """
    # Comprehensive input validation
    assert isinstance(pc_data, dict), f"pc_data must be dict, got {type(pc_data)}"
    assert 'pos' in pc_data, f"pc_data must contain 'pos' key, got keys: {list(pc_data.keys())}"
    assert key in pc_data, f"pc_data must contain '{key}' key, got keys: {list(pc_data.keys())}"
    
    points = pc_data['pos']  # Shape: (N, 3)
    labels = pc_data[key]   # Shape: (N,)
    
    # Validate points tensor
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"points must be 2D tensor with shape (N, 3), got shape {points.shape}"
    assert points.shape[1] == 3, f"points must have 3 coordinates (XYZ), got shape {points.shape}"
    assert points.shape[0] > 0, f"Point cloud cannot be empty, got {points.shape[0]} points"
    
    # Validate segmentation labels tensor
    assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
    assert labels.ndim == 1, f"labels must be 1D tensor with shape (N,), got shape {labels.shape}"
    assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
    
    # Validate camera matrices
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert camera_extrinsics.shape == (4, 4), f"camera_extrinsics must be 4x4 matrix, got shape {camera_extrinsics.shape}"
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (3, 3), f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    
    # Validate resolution
    assert isinstance(resolution, (tuple, list)), f"resolution must be tuple or list, got {type(resolution)}"
    assert len(resolution) == 2, f"resolution must have 2 elements (width, height), got {len(resolution)}"
    assert all(isinstance(x, int) and x > 0 for x in resolution), f"resolution must be positive integers, got {resolution}"
    
    # Validate convention
    assert isinstance(convention, str), f"convention must be str, got {type(convention)}"
    assert convention in ["opengl", "standard"], f"convention must be 'opengl' or 'standard', got '{convention}'"
    
    # Validate ignore_index
    assert isinstance(ignore_index, int), f"ignore_index must be int, got {type(ignore_index)}"
    assert 0 <= ignore_index <= 255, f"ignore_index must be in range [0, 255], got {ignore_index}"
    
    # Validate key parameter
    assert isinstance(key, str), f"key must be str, got {type(key)}"
    
    render_width, render_height = resolution
    
    # Ensure consistent device and dtype for precise calculations
    camera_extrinsics = camera_extrinsics.to(device=points.device, dtype=torch.float64)
    camera_intrinsics = camera_intrinsics.to(device=points.device, dtype=torch.float64)
    points = points.to(dtype=torch.float64)
    
    # Scale intrinsics matrix for target resolution
    # Estimate original resolution from principal point (cx, cy approximates image center)
    original_width = int(camera_intrinsics[0, 2] * 2)   # cx * 2 approximation
    original_height = int(camera_intrinsics[1, 2] * 2)  # cy * 2 approximation
    
    scale_x = render_width / original_width
    scale_y = render_height / original_height
    
    # Apply scaling to intrinsics matrix (clone to avoid modifying input)
    scaled_camera_intrinsics = camera_intrinsics.clone()
    scaled_camera_intrinsics[0, 0] *= scale_x  # Scale focal length fx
    scaled_camera_intrinsics[1, 1] *= scale_y  # Scale focal length fy
    scaled_camera_intrinsics[0, 2] *= scale_x  # Scale principal point cx
    scaled_camera_intrinsics[1, 2] *= scale_y  # Scale principal point cy
    
    # Transform points to camera coordinate system
    if convention == "opengl":
        # Convert to homogeneous coordinates for matrix transformation
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=torch.float64)
        points_homo = torch.cat([points, ones], dim=1)  # Shape: (N, 4)
        
        # Get world-to-camera transformation by inverting camera-to-world extrinsics
        world_to_camera = torch.inverse(camera_extrinsics)
        points_camera = (world_to_camera @ points_homo.T).T[:, :3]  # Shape: (N, 3)
        
        # Filter points in front of camera (OpenGL: camera looks down -Z axis)
        # Negative Z values indicate points in front of camera
        valid_depth = points_camera[:, 2] < 0
        points_3d = points_camera[valid_depth]
        valid_labels = labels[valid_depth]
    else:
        raise NotImplementedError("Standard convention not implemented yet")
    
    # Ensure we have points to render
    assert len(points_3d) > 0, f"No points in front of camera for segmentation rendering, got {len(points_3d)} valid points"
    
    # Project 3D camera coordinates to 2D image plane
    points_2d_homo = (scaled_camera_intrinsics @ points_3d.T).T  # Shape: (N, 3)
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # Perspective division, shape: (N, 2)
    depths = torch.abs(points_3d[:, 2])  # Distance from camera (absolute Z values)
    
    # Filter points within image bounds
    x_coords, y_coords = points_2d[:, 0], points_2d[:, 1]
    valid_x = (x_coords >= 0) & (x_coords < render_width)
    valid_y = (y_coords >= 0) & (y_coords < render_height)
    valid_points = valid_x & valid_y
    
    points_2d = points_2d[valid_points]
    depths = depths[valid_points]
    valid_labels = valid_labels[valid_points]
    
    # Ensure we have valid projections
    assert len(points_2d) > 0, f"No points within image bounds for segmentation rendering, got {len(points_2d)} valid points"
    
    # Initialize segmentation map with ignore values
    seg_map = torch.full((render_height, render_width), ignore_index, dtype=labels.dtype, device=labels.device)
    
    # Convert 2D coordinates to pixel indices with horizontal flip correction
    # X-coordinate flip corrects spatial alignment (fixes horizontal mirroring)
    pixel_x = (render_width - 1) - points_2d[:, 0].long()
    pixel_y = points_2d[:, 1].long()
    
    # Filter pixels within bounds after coordinate transformation
    valid_mask = (
        (pixel_x >= 0) & (pixel_x < render_width) &
        (pixel_y >= 0) & (pixel_y < render_height)
    )
    
    # Assign labels to valid pixels with depth ordering (closest point wins)
    if valid_mask.any():
        final_pixel_x = pixel_x[valid_mask]
        final_pixel_y = pixel_y[valid_mask]
        final_labels = valid_labels[valid_mask]
        final_depths = depths[valid_mask]
        
        # Sort by depth to handle overlapping projections (closest point wins)
        sorted_indices = torch.argsort(final_depths)
        sorted_pixel_x = final_pixel_x[sorted_indices]
        sorted_pixel_y = final_pixel_y[sorted_indices]
        sorted_labels = final_labels[sorted_indices]
        
        # Assign labels to pixels (first occurrence wins due to sorting)
        seg_map[sorted_pixel_y, sorted_pixel_x] = sorted_labels
    
    # Convert to int64 for final output
    return seg_map.to(dtype=torch.int64)
