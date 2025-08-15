"""Segmentation rendering from point clouds using projection."""

import torch
from typing import Dict, Tuple

from .coordinates import apply_coordinate_transform


def render_segmentation_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_extrinsics: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl",
    ignore_index: int = 255,
    key: str = "labels"
) -> torch.Tensor:
    """Render segmentation map from point cloud using Open3D.
    
    Args:
        pc_data: Point cloud dictionary with 'pos' and segmentation labels
        camera_extrinsics: 4x4 camera extrinsics matrix
        camera_intrinsics: 3x3 camera intrinsics matrix
        resolution: Target resolution as (width, height). Intrinsics are scaled automatically.
        convention: Camera extrinsics convention ("standard" or "opengl")
        ignore_index: Value to use for pixels with no point projection (default: 255)
        key: Key name for segmentation labels in pc_data (default: "labels")
        
    Returns:
        Segmentation map tensor [H, W] with integer labels
    """
    points = pc_data['pos']  # (N, 3)
    
    # CRITICAL: Point cloud must not be empty
    assert len(points) > 0, f"Point cloud cannot be empty, got {len(points)} points"
    
    # Validate input types - must be torch tensors
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    
    # CRITICAL: Point cloud data must contain segmentation labels - fail fast if not provided
    assert key in pc_data, f"Point cloud data must contain '{key}' key, got keys: {list(pc_data.keys())}"
    
    labels = pc_data[key]  # (N,)
    
    # CRITICAL: Labels must have correct shape - must be 1D
    assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
    assert labels.ndim == 1, f"labels must be 1D tensor with shape (N,), got shape {labels.shape}"
    assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
    
    # Extract target resolution
    render_width, render_height = resolution
        
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
    
    # Project 3D points to 2D image coordinates
    # Use the same approach as render_depth.py for OpenGL convention
    # transforms.json uses OpenGL convention: camera looks down -Z axis
    
    # Move to same device as points
    camera_extrinsics = camera_extrinsics.to(points.device)
    camera_intrinsics = scaled_intrinsics.to(points.device)
    
    if convention == "opengl":
        # Convert to homogeneous coordinates
        ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
        points_homo = torch.cat([points, ones], dim=1)
        
        # Transform to camera coordinates using standard matrix multiplication
        # This is exactly what depth rendering does
        world_to_camera = torch.inverse(camera_extrinsics)
        points_camera = (world_to_camera @ points_homo.T).T[:, :3]
        
        # Filter points behind camera (OpenGL convention: camera looks down -Z)
        # Negative Z values are in front of camera in OpenGL
        valid_depth = points_camera[:, 2] < 0
        points_3d = points_camera[valid_depth]
        valid_labels = labels[valid_depth]
    else:
        # Standard convention case (if needed)
        raise NotImplementedError("Standard convention not implemented yet")
    
    # CRITICAL: Must have points in front of camera for valid rendering
    assert len(points_3d) > 0, f"No points in front of camera for segmentation rendering, got {len(points_3d)} valid points"
    
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
    valid_labels = valid_labels[valid_points]
    
    # CRITICAL: Must have points within image bounds for valid segmentation map
    assert len(points_2d) > 0, f"No points within image bounds for segmentation rendering, got {len(points_2d)} valid points"
    
    # CRITICAL: Apply X-coordinate flip to correct spatial alignment (same as depth rendering)
    # This fixes the horizontal mirroring in the segmentation rendering
    pixel_x = (render_width - 1) - points_2d[:, 0].long()
    pixel_y = points_2d[:, 1].long()
    
    # Filter pixels that are within bounds after coordinate transformation
    valid_mask = (
        (pixel_x >= 0) & (pixel_x < render_width) &
        (pixel_y >= 0) & (pixel_y < render_height)
    )
    
    # Create segmentation map
    seg_map = torch.full((render_height, render_width), ignore_index, dtype=torch.long, device=points.device)
    
    # Assign labels to valid pixels with proper depth ordering (vectorized)
    if valid_mask.any():
        final_pixel_x = pixel_x[valid_mask]
        final_pixel_y = pixel_y[valid_mask]
        final_labels = valid_labels[valid_mask]
        final_depths = depths[valid_mask]
        
        # Sort by depth to get minimum depth per pixel (closest point wins)
        # This follows the exact same pattern as render_depth.py
        sorted_indices = torch.argsort(final_depths)
        sorted_pixel_x = final_pixel_x[sorted_indices]
        sorted_pixel_y = final_pixel_y[sorted_indices]
        sorted_labels = final_labels[sorted_indices]
        
        # Use index_put to assign labels - first occurrence wins (minimum depth)
        # This ensures that closer points override farther ones
        seg_map[sorted_pixel_y, sorted_pixel_x] = sorted_labels.long()
    
    return seg_map
