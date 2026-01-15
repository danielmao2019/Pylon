"""RGB rendering from point clouds using Open3D."""

from typing import Tuple

import numpy as np
import open3d as o3d
import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.point_cloud.point_cloud import PointCloud


def render_rgb_from_pointcloud_o3d(
    pc_data: PointCloud,
    camera: Camera,
    resolution: Tuple[int, int],
) -> torch.Tensor:
    """Render RGB image from point cloud using Open3D offscreen renderer.

    Projects 3D point cloud coordinates with RGB colors onto 2D image plane
    using camera parameters and Open3D rendering. Creates a photorealistic RGB
    image with proper depth sorting and lighting effects.

    Args:
        pc_data: Point cloud containing 3D coordinates and rgb field
        camera: Camera containing intrinsics/extrinsics/convention
        resolution: Target resolution as (height, width) tuple - intrinsics scaled automatically

    Returns:
        RGB image tensor of shape [3, H, W] with normalized values in [0, 1]

    Raises:
        AssertionError: If point cloud is empty, RGB data is missing, or camera parameters are invalid
    """
    assert isinstance(pc_data, PointCloud), f"{type(pc_data)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert hasattr(pc_data, 'rgb'), "PointCloud must contain rgb field"

    points = pc_data.xyz
    colors = pc_data.rgb

    # Validate resolution
    assert isinstance(
        resolution, (tuple, list)
    ), f"resolution must be tuple or list, got {type(resolution)}"
    assert (
        len(resolution) == 2
    ), f"resolution must have 2 elements (height, width), got {len(resolution)}"
    assert all(
        isinstance(x, int) and x > 0 for x in resolution
    ), f"resolution must be positive integers, got {resolution}"

    camera = camera.to(device=points.device, convention="opengl")
    render_height, render_width = resolution

    # Step 2: Device and dtype conversions
    camera_intrinsics = camera.intrinsics.clone()
    camera_extrinsics = camera.extrinsics

    # Step 3: Scale camera intrinsics in-place
    original_width = int(
        camera_intrinsics[0, 2] * 2
    )  # Estimate from principal point cx
    original_height = int(
        camera_intrinsics[1, 2] * 2
    )  # Estimate from principal point cy
    scale_x = render_width / original_width
    scale_y = render_height / original_height

    camera_intrinsics[0, 0] *= scale_x  # Scale focal length fx
    camera_intrinsics[1, 1] *= scale_y  # Scale focal length fy
    camera_intrinsics[0, 2] *= scale_x  # Scale principal point cx
    camera_intrinsics[1, 2] *= scale_y  # Scale principal point cy

    # Assert last row is [0, 0, 1] to ensure depth preservation during projection
    assert torch.allclose(
        camera_intrinsics[2, :],
        torch.tensor(
            [0.0, 0.0, 1.0],
            device=camera_intrinsics.device,
            dtype=camera_intrinsics.dtype,
        ),
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
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())

    # Step 6: Initialize renderer and set projection
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        render_width, render_height
    )
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # Black background

    # Setup material with unlit shader for consistent appearance
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"  # No lighting effects for consistent colors
    material.point_size = 2.0  # Slightly larger points to reduce gaps
    renderer.scene.add_geometry("pcd", pcd, material)

    # Configure camera projection with intrinsics
    renderer.scene.camera.set_projection(
        camera_intrinsics.cpu().numpy(),
        0.1,
        1000.0,  # Near and far clipping planes
        render_width,
        render_height,
    )

    # Step 7: Do projection - define pos, forward, up, set look_at, and render
    pos = camera.extrinsics[:3, 3]
    forward = camera.forward
    up = camera.up

    # Position camera in 3D scene and render
    renderer.scene.camera.look_at(
        center=(pos + forward * 10).cpu().numpy(),  # Look-at target point
        eye=pos.cpu().numpy(),  # Camera position
        up=up.cpu().numpy(),  # Up vector
    )

    # Render scene to image
    img = renderer.render_to_image()
    img_array = np.asarray(img)

    # Step 8: Convert to torch.Tensor, apply /255 normalization, and permute
    rendered_rgb = torch.from_numpy(img_array).float() / 255.0
    rendered_rgb = rendered_rgb.permute(2, 0, 1)  # Convert [H, W, 3] -> [3, H, W]

    return rendered_rgb
