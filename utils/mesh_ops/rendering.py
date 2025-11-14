"""Rendering helpers for triangle meshes using PyTorch3D."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes

from utils.input_checks.check_camera import (
    check_camera_extrinsics,
    check_camera_intrinsics,
)
from utils.three_d.camera.conventions import apply_coordinate_transform
from utils.three_d.camera.scaling import scale_intrinsics


def _prepare_cameras(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str,
    device: torch.device,
) -> PerspectiveCameras:
    intrinsics = check_camera_intrinsics(intrinsics).clone().to(device=device)
    extrinsics = check_camera_extrinsics(extrinsics).clone().to(device=device)

    scale_intrinsics(intrinsics=intrinsics, resolution=resolution, inplace=True)

    # Convert to PyTorch3D coordinate system (right-handed):
    # - X: left (+X points left)
    # - Y: up (+Y points up)
    # - Z: forward (+Z points from us to scene, out from image plane)
    extrinsics_p3d = apply_coordinate_transform(
        extrinsics=extrinsics,
        source_convention=convention,
        target_convention='pytorch3d',
    )

    rotation_c2w = extrinsics_p3d[:3, :3]
    translation_c2w = extrinsics_p3d[:3, 3]

    rotation_w2c = rotation_c2w
    translation_w2c = -rotation_c2w.transpose(0, 1) @ translation_c2w

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    image_height, image_width = resolution

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor(
            [[image_height, image_width]], dtype=torch.float32, device=device
        ),
        R=rotation_w2c.unsqueeze(0).to(dtype=torch.float32),
        T=translation_w2c.unsqueeze(0).to(dtype=torch.float32),
        in_ndc=False,
    )

    # Set zfar to maximum float32 value to prevent shader overflow with distant geometry
    # PyTorch3D's SoftPhongShader uses default zfar=100.0 which causes black
    # output for meshes beyond that depth due to arithmetic overflow in blending
    # Use maximum float32 value (not inf or float64 max) to match PyTorch3D's internal precision
    cameras.zfar = float(np.finfo(np.float32).max)

    return cameras.to(device=device)


def _build_rasterizer(
    cameras: PerspectiveCameras,
    resolution: Tuple[int, int],
) -> MeshRasterizer:
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
        bin_size=0,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def _build_shader(
    cameras: PerspectiveCameras,
    device: torch.device,
    background_color: Tuple[int, int, int],
) -> SoftPhongShader:
    assert len(background_color) == 3
    assert all(isinstance(channel, int) for channel in background_color)
    assert all(0 <= channel <= 255 for channel in background_color)
    background = tuple(float(channel) / 255.0 for channel in background_color)

    lights = PointLights(
        device=device,
        location=torch.zeros(1, 3, device=device),
        ambient_color=torch.ones(1, 3, device=device),
        diffuse_color=torch.zeros(1, 3, device=device),
        specular_color=torch.zeros(1, 3, device=device),
    )

    blend_params = BlendParams(background_color=background)

    return SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params,
    )


@torch.no_grad()
def render_rgb_from_mesh(
    meshes: Meshes,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = 'opengl',
    background: Tuple[int, int, int] = (0, 0, 0),
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render an RGB image from a triangle mesh using PyTorch3D.

    Args:
        mesh_data: Meshes object produced by `load_meshes`.
        intrinsics: 3x3 camera intrinsics matrix.
        extrinsics: 4x4 camera-to-world transform for the camera.
        resolution: Target image size (height, width).
        convention: Coordinate convention of `extrinsics` (default: 'opengl').
        background: Background RGB color as integer tuple in [0, 255].
        return_mask: If True, also return valid pixel mask (default: False).

    Returns:
        If return_mask is False:
            RGB image tensor of shape [3, H, W] in range [0, 1].
        If return_mask is True:
            Tuple of (RGB image tensor, valid mask tensor of shape [H, W]).
    """
    assert isinstance(resolution, tuple) and len(resolution) == 2
    assert torch.cuda.is_available(), "CUDA device required for mesh rendering"

    device = torch.device('cuda')
    assert isinstance(meshes, Meshes)
    meshes = meshes.clone().to(device=device)

    intrinsics = intrinsics.to(device=device)
    extrinsics = extrinsics.to(device=device)

    cameras = _prepare_cameras(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        resolution=resolution,
        convention=convention,
        device=device,
    )

    rasterizer = _build_rasterizer(cameras=cameras, resolution=resolution)
    shader = _build_shader(cameras=cameras, device=device, background_color=background)

    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # Render RGB image (shape: [batch_size, H, W, 4])
    images = renderer(meshes)

    # Extract RGB channels [3, H, W]
    rgb = images[0, :, :, :3].permute(2, 0, 1).contiguous()
    rgb = rgb.clamp(0.0, 1.0)

    # Handle mask creation if requested
    if return_mask:
        # Get rasterization fragments to determine mesh presence
        # Rasterizer returns fragments with pix_to_face indicating which face each pixel belongs to
        # pix_to_face = -1 means no mesh geometry at that pixel
        fragments = rasterizer(meshes)

        # Create binary mask: True where mesh exists, False for background
        # fragments.pix_to_face has shape [batch_size, H, W, faces_per_pixel]
        # We only check the closest face (index 0)
        valid_mask = fragments.pix_to_face[0, :, :, 0] >= 0

        return rgb, valid_mask
    else:
        return rgb
