"""RGB rendering from point clouds using Open3D."""

import torch
import numpy as np
import open3d as o3d
from typing import Dict, Tuple

from .coordinates import apply_coordinate_transform


def render_rgb_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_extrinsics: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl"
) -> torch.Tensor:
    """Render RGB image from point cloud using Open3D offscreen renderer.
    
    Creates a photorealistic RGB image by projecting 3D point cloud data onto
    a 2D image plane using camera parameters. Supports both OpenGL and standard
    camera conventions with automatic intrinsics scaling for target resolution.
    
    Args:
        pc_data: Point cloud dictionary containing 'pos' key with 3D coordinates
                 and 'rgb' key with color information (mandatory)
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world transform)
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        resolution: Target resolution as (width, height) tuple - intrinsics scaled automatically
        convention: Camera extrinsics convention ("opengl" or "standard")
        
    Returns:
        RGB image tensor of shape [3, H, W] with normalized values in [0, 1]
        
    Raises:
        AssertionError: If point cloud is empty, RGB data is missing, or camera parameters are invalid
    """
    # Comprehensive input validation
    assert isinstance(pc_data, dict), f"pc_data must be dict, got {type(pc_data)}"
    assert 'pos' in pc_data, f"pc_data must contain 'pos' key, got keys: {list(pc_data.keys())}"
    assert 'rgb' in pc_data, f"pc_data must contain 'rgb' key, got keys: {list(pc_data.keys())}"
    
    points = pc_data['pos']  # Shape: (N, 3)
    colors = pc_data['rgb']  # Shape: (N, 3)
    
    # Validate points tensor
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"points must be 2D tensor with shape (N, 3), got shape {points.shape}"
    assert points.shape[1] == 3, f"points must have 3 coordinates (XYZ), got shape {points.shape}"
    assert points.shape[0] > 0, f"Point cloud cannot be empty, got {points.shape[0]} points"
    
    # Validate RGB colors tensor
    assert isinstance(colors, torch.Tensor), f"colors must be torch.Tensor, got {type(colors)}"
    assert colors.ndim == 2, f"colors must be 2D tensor with shape (N, 3), got shape {colors.shape}"
    assert colors.shape[1] == 3, f"colors must have 3 channels (RGB), got shape {colors.shape}"
    assert colors.shape[0] == points.shape[0], f"colors length {colors.shape[0]} != points length {points.shape[0]}"
    
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
    
    render_width, render_height = resolution
    
    # Scale intrinsics matrix for target resolution
    # Estimate original resolution from principal point (cx, cy approximates image center)
    original_width = int(camera_intrinsics[0, 2] * 2)   # cx * 2 approximation
    original_height = int(camera_intrinsics[1, 2] * 2)  # cy * 2 approximation
    
    scale_x = render_width / original_width
    scale_y = render_height / original_height
    
    # Apply scaling to intrinsics matrix
    scaled_intrinsics = camera_intrinsics.clone()
    scaled_intrinsics[0, 0] *= scale_x  # Scale focal length fx
    scaled_intrinsics[1, 1] *= scale_y  # Scale focal length fy
    scaled_intrinsics[0, 2] *= scale_x  # Scale principal point cx
    scaled_intrinsics[1, 2] *= scale_y  # Scale principal point cy
    
    camera_intrinsics = scaled_intrinsics
    
    # Create Open3D point cloud with double precision for maximum accuracy
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().double().numpy())
    
    # Set RGB colors (already validated above)
    # Check if normalization is needed BEFORE converting to numpy
    integer_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    is_integer_dtype = colors.dtype in integer_dtypes
    is_in_255_range = colors.min() >= 0 and colors.max() <= 255 and colors.max() > 1.0
    
    # Normalize if needed, then convert to numpy for Open3D
    if is_integer_dtype and is_in_255_range:
        colors = colors / 255.0
    
    colors = colors.cpu().double().numpy()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Initialize offscreen renderer
    render_size = (render_width, render_height)
    renderer = o3d.visualization.rendering.OffscreenRenderer(render_size[0], render_size[1])
    
    # Configure rendering settings
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black background
    
    # Setup material with unlit shader for consistent appearance
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"  # No lighting effects for consistent colors
    material.point_size = 2.0         # Slightly larger points to reduce gaps
    
    renderer.scene.add_geometry("pcd", pcd, material)
    
    # Configure camera projection with intrinsics
    renderer.scene.camera.set_projection(
        camera_intrinsics.cpu().double().numpy(), 
        0.1, 1000.0,  # Near and far clipping planes
        render_size[0], render_size[1]
    )
    
    # Extract camera pose from camera-to-world extrinsics matrix
    if convention == "opengl":
        # camera_extrinsics is already camera-to-world transform
        pos = camera_extrinsics[:3, 3]        # Camera position in world coordinates
        forward = -camera_extrinsics[:3, 2]   # Forward direction (-Z in OpenGL)
        up = camera_extrinsics[:3, 1]         # Up direction (Y in OpenGL)
    else:
        # Standard convention: transform to OpenGL coordinate system
        transformed_extrinsics = apply_coordinate_transform(
            camera_extrinsics, 
            source_convention="standard", 
            target_convention="opengl"
        )
        # transformed_extrinsics is camera-to-world transform
        pos = transformed_extrinsics[:3, 3]        # Camera position in world coordinates
        forward = -transformed_extrinsics[:3, 2]   # Forward direction (-Z in OpenGL)
        up = transformed_extrinsics[:3, 1]         # Up direction (Y in OpenGL)
    
    # Position camera in 3D scene
    renderer.scene.camera.look_at(
        center=(pos + forward * 10).cpu().double().numpy(),  # Look-at target point
        eye=pos.cpu().double().numpy(),                      # Camera position
        up=up.cpu().double().numpy(),                        # Up vector
    )
    
    # Render scene to image
    img = renderer.render_to_image()
    img_array = np.asarray(img)
    
    # Convert to PyTorch tensor with normalized values
    rendered_rgb = torch.from_numpy(img_array).float() / 255.0
    rendered_rgb = rendered_rgb.permute(2, 0, 1)  # Convert [H, W, 3] -> [3, H, W]
    
    return rendered_rgb
