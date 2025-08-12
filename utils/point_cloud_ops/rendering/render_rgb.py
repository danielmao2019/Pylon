"""RGB rendering from point clouds using Open3D."""

import torch
import numpy as np
import open3d as o3d
from typing import Dict, Union, Tuple

from .coordinates import apply_coordinate_transform


def render_rgb_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_extrinsics: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "standard"
) -> torch.Tensor:
    """Render RGB image from point cloud using Open3D.
    
    Args:
        pc_data: Point cloud dictionary with 'pos' and optionally 'rgb' keys
        camera_extrinsics: 4x4 camera extrinsics matrix
        camera_intrinsics: 3x3 camera intrinsics matrix
        resolution: Target resolution as (width, height). Intrinsics are scaled automatically.
        convention: Camera extrinsics convention ("standard" or "opengl")
        
    Returns:
        RGB image tensor [3, H, W] with values in [0, 1]
    """
    points = pc_data['pos']  # (N, 3)
    
    # CRITICAL: Point cloud must not be empty
    assert len(points) > 0, f"Point cloud cannot be empty, got {len(points)} points"
    
    # Validate input types - must be torch tensors
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    
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
    
    camera_intrinsics = scaled_intrinsics
    
    # For OpenGL convention (transforms.json), work directly in OpenGL coordinates
    # Open3D renderer expects OpenGL convention natively
    # Convert to numpy only when needed for Open3D API
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    
    # Set colors if available
    if 'rgb' in pc_data:
        colors = pc_data['rgb'].cpu().numpy()
        # Normalize colors to [0, 1] if needed
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Setup renderer
    render_size = (render_width, render_height)
    renderer = o3d.visualization.rendering.OffscreenRenderer(render_size[0], render_size[1])
    
    # Set background to black to avoid brightness from default gray/white background
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black background
    
    # Setup material with unlit shader (no lighting effects)
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0  # Slightly larger points to reduce gaps
    
    renderer.scene.add_geometry("pcd", pcd, material)
    
    # Set camera projection (convert to numpy only for Open3D API)
    renderer.scene.camera.set_projection(
        camera_intrinsics.cpu().numpy(), 
        0.1, 1000.0, 
        render_size[0], render_size[1]
    )
    
    # Set camera view based on convention
    if convention == "opengl":
        # transforms.json uses OpenGL convention (camera-to-world transform)
        # Extract camera parameters from extrinsics matrix directly
        pos = camera_extrinsics[:3, 3]        # Camera position in world
        forward = -camera_extrinsics[:3, 2]   # Forward is -Z (OpenGL convention)
        up = camera_extrinsics[:3, 1]         # Up is Y (OpenGL convention)
    else:
        # Standard convention: transform to OpenGL first
        transformed_extrinsics = apply_coordinate_transform(
            camera_extrinsics, 
            source_convention="standard", 
            target_convention="opengl"
        )
        # Extract camera parameters from transformed extrinsics matrix
        pos = transformed_extrinsics[:3, 3]        # Camera position in world
        forward = -transformed_extrinsics[:3, 2]   # Forward is -Z (OpenGL convention)
        up = transformed_extrinsics[:3, 1]         # Up is Y (OpenGL convention)
    
    # Set camera view (convert to numpy only at API call)
    renderer.scene.camera.look_at(
        center=(pos + forward * 10).cpu().numpy(),  # Convert only at API call
        eye=pos.cpu().numpy(),                      # Convert only at API call
        up=up.cpu().numpy(),                        # Convert only at API call
    )
    
    # Render and convert to tensor
    img = renderer.render_to_image()
    img_array = np.asarray(img)
    
    # Convert to tensor [3, H, W] with values in [0, 1]
    rendered_rgb = torch.from_numpy(img_array).float() / 255.0
    rendered_rgb = rendered_rgb.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    
    return rendered_rgb
