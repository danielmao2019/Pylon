"""RGB rendering from point clouds using Open3D."""

import torch
import numpy as np
import open3d as o3d
from typing import Dict, Tuple

from .coordinates import apply_coordinate_transform


def render_rgb_from_pointcloud(
    pc_data: Dict[str, torch.Tensor],
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = "opengl"
) -> torch.Tensor:
    """Render RGB image from point cloud using Open3D offscreen renderer.

    Projects 3D point cloud coordinates with RGB colors onto 2D image plane
    using camera parameters and Open3D rendering. Creates a photorealistic RGB
    image with proper depth sorting and lighting effects.

    Args:
        pc_data: Point cloud dictionary containing 'pos' key with 3D coordinates
                 and 'rgb' key with color information (mandatory)
        camera_intrinsics: 3x3 camera intrinsics matrix with focal lengths and principal point
        camera_extrinsics: 4x4 camera extrinsics matrix (camera-to-world transform)
        resolution: Target resolution as (width, height) tuple - intrinsics scaled automatically
        convention: Camera extrinsics convention ("opengl" or "standard")

    Returns:
        RGB image tensor of shape [3, H, W] with normalized values in [0, 1]

    Raises:
        AssertionError: If point cloud is empty, RGB data is missing, or camera parameters are invalid
    """
    # Step 1: Input validation
    assert isinstance(pc_data, dict), f"pc_data must be dict, got {type(pc_data)}"
    assert 'pos' in pc_data, f"pc_data must contain 'pos' key, got keys: {list(pc_data.keys())}"
    assert 'rgb' in pc_data, f"pc_data must contain 'rgb' key, got keys: {list(pc_data.keys())}"

    points = pc_data['pos']
    colors = pc_data['rgb']

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
    assert isinstance(camera_intrinsics, torch.Tensor), f"camera_intrinsics must be torch.Tensor, got {type(camera_intrinsics)}"
    assert camera_intrinsics.shape == (3, 3), f"camera_intrinsics must be 3x3 matrix, got shape {camera_intrinsics.shape}"
    assert isinstance(camera_extrinsics, torch.Tensor), f"camera_extrinsics must be torch.Tensor, got {type(camera_extrinsics)}"
    assert camera_extrinsics.shape == (4, 4), f"camera_extrinsics must be 4x4 matrix, got shape {camera_extrinsics.shape}"

    # Validate all tensors are on same device
    assert points.device == camera_intrinsics.device, f"points device {points.device} != camera_intrinsics device {camera_intrinsics.device}"
    assert points.device == camera_extrinsics.device, f"points device {points.device} != camera_extrinsics device {camera_extrinsics.device}"
    assert points.device == colors.device, f"points device {points.device} != colors device {colors.device}"

    # Validate resolution
    assert isinstance(resolution, (tuple, list)), f"resolution must be tuple or list, got {type(resolution)}"
    assert len(resolution) == 2, f"resolution must have 2 elements (width, height), got {len(resolution)}"
    assert all(isinstance(x, int) and x > 0 for x in resolution), f"resolution must be positive integers, got {resolution}"

    # Validate convention
    assert isinstance(convention, str), f"convention must be str, got {type(convention)}"
    assert convention in ["opengl", "standard"], f"convention must be 'opengl' or 'standard', got '{convention}'"

    render_width, render_height = resolution

    # Step 2: Device and dtype conversions
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

    # Step 4: Handle color normalization
    # Check if normalization is needed and apply /255 if required
    integer_dtypes = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
    is_integer_dtype = colors.dtype in integer_dtypes
    is_in_255_range = colors.min() >= 0 and colors.max() <= 255 and colors.max() > 1.0

    # Normalize colors if needed
    if is_integer_dtype and is_in_255_range:
        colors = colors / 255.0

    # Step 5: Prepare point cloud data for Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().double().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().double().numpy())

    # Step 6: Initialize renderer and set projection
    renderer = o3d.visualization.rendering.OffscreenRenderer(render_width, render_height)
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
        render_width, render_height
    )

    # Step 7: Do projection - define pos, forward, up, set look_at, and render
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

    # Position camera in 3D scene and render
    renderer.scene.camera.look_at(
        center=(pos + forward * 10).cpu().double().numpy(),  # Look-at target point
        eye=pos.cpu().double().numpy(),                      # Camera position
        up=up.cpu().double().numpy(),                        # Up vector
    )
    
    # Render scene to image
    img = renderer.render_to_image()
    img_array = np.asarray(img)

    # Step 8: Convert to torch.Tensor, apply /255 normalization, and permute
    rendered_rgb = torch.from_numpy(img_array).float() / 255.0
    rendered_rgb = rendered_rgb.permute(2, 0, 1)  # Convert [H, W, 3] -> [3, H, W]

    return rendered_rgb
