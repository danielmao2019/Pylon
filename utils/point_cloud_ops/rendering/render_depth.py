"""Depth rendering from point clouds using projection methods."""

import torch
import numpy as np
from typing import Dict, Union, Tuple


def render_depth_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_extrinsics: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_index: float = -1.0
) -> torch.Tensor:
    """Render depth map from point cloud using projection.
    
    Args:
        pc_data: Point cloud dictionary with 'pos' key
        camera_extrinsics: 4x4 camera extrinsics matrix
        camera_intrinsics: 3x3 camera intrinsics matrix
        resolution: Target resolution as (width, height). Intrinsics are scaled automatically.
        convention: Camera extrinsics convention ("standard" or "opengl")
        ignore_index: Value for pixels with no corresponding points
        
    Returns:
        Depth map tensor [H, W] with depth values
    """
    points = pc_data['pos']  # (N, 3)
    
    # CRITICAL: Point cloud must not be empty
    assert len(points) > 0, f"Point cloud cannot be empty, got {len(points)} points"
    
    # Validate input types - must be torch tensors
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    
    # Extract target resolution
    render_width, render_height = resolution
    
    # Move to same device and dtype as points for precise calculations
    camera_extrinsics = camera_extrinsics.to(device=points.device, dtype=points.dtype)
    camera_intrinsics = camera_intrinsics.to(device=points.device, dtype=points.dtype)
    
    # Scale intrinsics for target resolution
    # Extract original resolution from intrinsics (assume standard format)
    # Principal point gives us the original image center
    original_width = int(camera_intrinsics[0, 2] * 2)  # cx * 2 approximation
    original_height = int(camera_intrinsics[1, 2] * 2)  # cy * 2 approximation
    
    # Calculate scale factors
    scale_x = render_width / original_width
    scale_y = render_height / original_height
    
    # Scale intrinsics matrix
    scaled_intrinsics = camera_intrinsics.clone()
    scaled_intrinsics[0, 0] *= scale_x  # fx
    scaled_intrinsics[1, 1] *= scale_y  # fy
    scaled_intrinsics[0, 2] *= scale_x  # cx
    scaled_intrinsics[1, 2] *= scale_y  # cy
    
    camera_intrinsics = scaled_intrinsics
    
    # Use the same approach as RGB rendering for OpenGL convention
    # transforms.json uses OpenGL convention: camera looks down -Z axis
    if convention == "opengl":
        # Convert to homogeneous coordinates
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=1)
        
        # Transform to camera coordinates using standard matrix multiplication
        # This is exactly what RGB rendering does internally
        world_to_camera = torch.inverse(camera_extrinsics)
        points_camera = (world_to_camera @ points_homo.T).T[:, :3]
        
        # Filter points behind camera (OpenGL convention: camera looks down -Z)
        # Negative Z values are in front of camera in OpenGL
        valid_depth = points_camera[:, 2] < 0
        points_3d = points_camera[valid_depth]
    else:
        # Standard convention case (if needed)
        raise NotImplementedError("Standard convention not implemented yet")
    
    # CRITICAL: Must have points in front of camera for valid rendering
    assert len(points_3d) > 0, f"No points in front of camera for depth rendering, got {len(points_3d)} valid points"
    
    # Project to image plane using OpenGL coordinates
    points_2d_homo = (camera_intrinsics @ points_3d.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    depths = torch.abs(points_3d[:, 2])  # Use absolute Z values (distance from camera)
    
    # Filter points within image bounds
    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]
    valid_x = (x_coords >= 0) & (x_coords < render_width)
    valid_y = (y_coords >= 0) & (y_coords < render_height)
    valid_points = valid_x & valid_y
    
    points_2d = points_2d[valid_points]
    depths = depths[valid_points]
    
    # Initialize depth map
    depth_map = torch.full(
        (render_height, render_width), 
        ignore_index, 
        dtype=torch.float32,
        device=points.device
    )
    
    # CRITICAL: Must have points within image bounds for valid depth map
    assert len(points_2d) > 0, f"No points within image bounds for depth rendering, got {len(points_2d)} valid points"
    
    # Fill depth map using vectorized approach
    # CRITICAL: Apply X-coordinate flip to correct spatial alignment
    # This fixes the horizontal mirroring in the depth rendering
    pixel_x = (render_width - 1) - points_2d[:, 0].long()
    pixel_y = points_2d[:, 1].long()
    
    # Sort by depth to get minimum depth per pixel (closest point wins)
    sorted_indices = torch.argsort(depths)
    sorted_pixel_x = pixel_x[sorted_indices]
    sorted_pixel_y = pixel_y[sorted_indices]
    sorted_depths = depths[sorted_indices]
    
    # Use index_put to assign depths - first occurrence wins (minimum depth)
    # Convert depths to float32 for final output while maintaining calculation precision
    depth_map[sorted_pixel_y, sorted_pixel_x] = sorted_depths.float()
    
    return depth_map
