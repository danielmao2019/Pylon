"""Rendering helpers for triangle meshes using PyTorch3D."""

import math
from typing import Tuple, Union

import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures import Meshes

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.mesh.convert import mesh_to_pytorch3d
from data.structures.three_d.mesh.mesh import Mesh

# SoftRas blend gamma shared by the silhouette blend params and the rasterizer
# blur radius, so the two stay coupled to a single PyTorch3D convention.
_SILHOUETTE_BLEND_GAMMA = 1e-4


def _prepare_cameras(
    camera: Camera,
    resolution: Tuple[int, int],
    device: torch.device,
) -> PerspectiveCameras:
    assert isinstance(camera, Camera), f"{type(camera)=}"
    camera_prepared = camera.to(device=device, convention='pytorch3d').scale_intrinsics(
        resolution=resolution
    )
    intrinsics = camera_prepared.intrinsics

    # Convert to PyTorch3D coordinate system (right-handed):
    # - X: left (+X points left)
    # - Y: up (+Y points up)
    # - Z: forward (+Z points from us to scene, out from image plane)
    rotation_w2c_col = camera_prepared.extrinsics.w2c[:3, :3]
    translation_w2c_col = camera_prepared.extrinsics.w2c[:3, 3]

    # PyTorch3D expects row-major world-to-camera: X_cam = X_world @ R + T
    rotation_w2c = rotation_w2c_col.transpose(0, 1)
    translation_w2c = translation_w2c_col

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.cx
    cy = intrinsics.cy

    image_height, image_width = resolution

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor(
            [[image_height, image_width]], dtype=torch.float32, device=device
        ),
        R=rotation_w2c.unsqueeze(0),
        T=translation_w2c.unsqueeze(0),
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


def _build_soft_rasterizer(
    cameras: CamerasBase,
    resolution: Tuple[int, int],
    blur_sigma: float,
    faces_per_pixel: int,
) -> MeshRasterizer:
    """Build a blurred multi-sample MeshRasterizer whose blur radius and faces-per-pixel realize the SoftRas probabilistic coverage.

    Args:
        cameras: PyTorch3D cameras the rasterizer projects through.
        resolution: Target image size (height, width).
        blur_sigma: SoftRas blend sigma; the rasterization blur radius is derived from it as log(1 / gamma - 1) * blur_sigma with gamma = 1e-4 (PyTorch3D convention).
        faces_per_pixel: Number of faces rasterized per pixel for the coverage blend.

    Returns:
        MeshRasterizer with the derived blur radius and the given faces-per-pixel.
    """
    # PyTorch3D SoftRas convention: derive the rasterization blur radius from the
    # blend sigma and gamma, matching the official silhouette examples.
    blur_radius = math.log(1.0 / _SILHOUETTE_BLEND_GAMMA - 1.0) * blur_sigma
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def _build_silhouette_shader(blur_sigma: float) -> SoftSilhouetteShader:
    """Build the PyTorch3D SoftSilhouetteShader that aggregates per-face coverage probabilities under blur_sigma.

    Args:
        blur_sigma: SoftRas blend sigma controlling the sigmoid softness of the coverage aggregation.

    Returns:
        SoftSilhouetteShader blending with sigma set to blur_sigma and gamma = 1e-4.
    """
    return SoftSilhouetteShader(
        blend_params=BlendParams(sigma=blur_sigma, gamma=_SILHOUETTE_BLEND_GAMMA)
    )


@torch.no_grad()
def render_rgb_from_mesh(
    mesh: Mesh,
    camera: Camera,
    resolution: Tuple[int, int],
    background: Tuple[int, int, int] = (0, 0, 0),
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render an RGB image from a triangle mesh using PyTorch3D.

    Args:
        mesh: Repo `Mesh` object to render.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target image size (height, width).
        background: Background RGB color as integer tuple in [0, 255].
        return_mask: If True, also return valid pixel mask (default: False).

    Returns:
        If return_mask is False:
            RGB image tensor of shape [3, H, W] in range [0, 1].
        If return_mask is True:
            Tuple of (RGB image tensor, valid mask tensor of shape [H, W]).
    """
    assert isinstance(resolution, tuple) and len(resolution) == 2
    assert isinstance(mesh, Mesh), f"{type(mesh)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"

    assert torch.cuda.is_available(), "CUDA device required for mesh rendering"
    device = torch.device('cuda')
    meshes = mesh_to_pytorch3d(
        mesh=mesh,
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(meshes, Meshes), f"{type(meshes)=}"

    cameras = _prepare_cameras(
        camera=camera,
        resolution=resolution,
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


def render_soft_silhouette_from_mesh(
    mesh: Mesh,
    camera: Camera,
    resolution: Tuple[int, int],
    blur_sigma: float,
    faces_per_pixel: int = 50,
) -> torch.Tensor:
    """Render a differentiable SoftRas-style soft silhouette from a triangle mesh.

    Renders the per-pixel probability that a pixel is covered by the mesh using
    PyTorch3D's `SoftSilhouetteShader`, which sigmoid-blends the rasterized
    face-to-pixel distances with softness set by `blur_sigma`. Unlike
    `render_rgb_from_mesh`, this path is NOT wrapped in `torch.no_grad()`, so a
    soft-IoU silhouette loss can back-propagate into `mesh.verts` and the camera
    parameters. Coarse-to-fine annealing is the caller's job via `blur_sigma`.

    Args:
        mesh: Repo `Mesh` object to render; its `verts` `[V, 3]` float32 tensor
            carries the gradient path.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Target image size (height, width) as positive ints.
        blur_sigma: SoftRas blend sigma controlling silhouette edge softness. The
            rasterizer blur radius is derived from it as
            log(1 / gamma - 1) * blur_sigma with gamma = 1e-4 (PyTorch3D
            convention). Larger sigma yields a blurrier/coarser silhouette.
        faces_per_pixel: Number of faces blended per pixel by the rasterizer
            (default: 50).

    Returns:
        Soft silhouette tensor of shape [H, W] (H, W = resolution), float32 in
        [0, 1], differentiable w.r.t. `mesh.verts` and the camera parameters.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(camera, Camera), f"{type(camera)=}"
        assert isinstance(resolution, tuple) and len(resolution) == 2, (
            "Expected `resolution` to be a (height, width) tuple of length 2. "
            f"{resolution=}"
        )
        assert all(isinstance(dim, int) and dim > 0 for dim in resolution), (
            "Expected `resolution` values to be positive integers. " f"{resolution=}"
        )
        assert isinstance(blur_sigma, float) and blur_sigma > 0.0, (
            "Expected `blur_sigma` to be a positive float. " f"{blur_sigma=}"
        )
        assert isinstance(faces_per_pixel, int) and faces_per_pixel > 0, (
            "Expected `faces_per_pixel` to be a positive integer. "
            f"{faces_per_pixel=}"
        )

    _validate_inputs()

    assert torch.cuda.is_available(), "CUDA device required for mesh rendering"
    device = torch.device('cuda')

    meshes = mesh_to_pytorch3d(
        mesh=mesh,
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(meshes, Meshes), f"{type(meshes)=}"

    cameras = _prepare_cameras(
        camera=camera,
        resolution=resolution,
        device=device,
    )

    rasterizer = _build_soft_rasterizer(
        cameras=cameras,
        resolution=resolution,
        blur_sigma=blur_sigma,
        faces_per_pixel=faces_per_pixel,
    )
    shader = _build_silhouette_shader(blur_sigma=blur_sigma)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    # SoftSilhouetteShader output is [batch_size, H, W, 4]; the alpha channel
    # (index 3) holds the soft silhouette coverage probability in [0, 1].
    images = renderer(meshes)
    silhouette = images[0, :, :, 3].contiguous()

    return silhouette
