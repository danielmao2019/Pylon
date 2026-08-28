"""Rendering helpers for triangle meshes using PyTorch3D."""

from math import log
from typing import Optional, Tuple, Union

import numpy as np
import torch
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    OrthographicCameras,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.mesh.convert import mesh_to_pytorch3d
from data.structures.three_d.mesh.mesh import Mesh


def _prepare_cameras(
    camera: Camera,
    resolution: Tuple[int, int],
    device: torch.device,
) -> CamerasBase:
    """Build the PyTorch3D camera used by mesh rasterization.

    Args:
        camera: Repo Camera whose intrinsics and extrinsics may use any supported convention.
        resolution: Target image size `(height, width)` for restating intrinsics.
        device: Torch device where PyTorch3D should allocate camera tensors.

    Returns:
        PyTorch3D camera in PyTorch3D's pose and image-plane frame.
    """

    def _validate_inputs() -> None:
        assert isinstance(
            camera, Camera
        ), f"Expected `camera` to be a Camera. {type(camera)=}"
        assert isinstance(resolution, tuple) and len(resolution) == 2, (
            "Expected `resolution` to be a length-2 tuple. " f"{resolution=}"
        )
        for axis_name, value in zip(("height", "width"), resolution, strict=True):
            assert isinstance(value, int) and not isinstance(value, bool), (
                "Expected each render resolution value to be an int. "
                f"{axis_name=} {type(value)=} {resolution=}"
            )
            assert value > 0, (
                "Expected render resolution values to be positive integers. "
                f"{axis_name=} {value=} {resolution=}"
            )
        assert isinstance(device, torch.device), (
            "Expected `device` to be a torch.device. " f"{type(device)=}"
        )

    _validate_inputs()

    def _normalize_inputs(camera: Camera) -> Camera:
        camera = camera.to(
            device=device,
            dtype=torch.float32,
            intr_convention="pytorch3d",
            extr_convention="pytorch3d",
        ).scale_intrinsics(resolution=resolution)
        return camera

    camera = _normalize_inputs(camera=camera)

    intrinsics = camera.intrinsics

    rotation_w2c_col = camera.extrinsics.w2c[:3, :3]
    translation_w2c_col = camera.extrinsics.w2c[:3, 3]

    # PyTorch3D expects row-major world-to-camera: X_cam = X_world @ R + T
    rotation_w2c = rotation_w2c_col.transpose(0, 1)
    translation_w2c = translation_w2c_col

    focal_length = torch.stack((intrinsics.fx, intrinsics.fy)).reshape(1, 2)
    principal_point = torch.stack((intrinsics.cx, intrinsics.cy)).reshape(1, 2)

    if intrinsics.model in {"simple_pinhole", "pinhole"}:
        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=rotation_w2c.unsqueeze(0),
            T=translation_w2c.unsqueeze(0),
            in_ndc=True,
            device=device,
        )
        cameras.zfar = float(np.finfo(np.float32).max)
        return cameras.to(device=device)

    if intrinsics.model == "ortho":
        cameras = OrthographicCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=rotation_w2c.unsqueeze(0),
            T=translation_w2c.unsqueeze(0),
            in_ndc=True,
            device=device,
        )
        cameras.zfar = float(np.finfo(np.float32).max)
        return cameras.to(device=device)

    assert False, "Should not reach here. " f"{intrinsics.model=}"


def _build_rasterizer(
    cameras: CamerasBase,
    resolution: Tuple[int, int],
    blur_radius: float = 0.0,
    faces_per_pixel: int = 1,
    clip_barycentric_coords: bool = False,
) -> MeshRasterizer:
    """Build a PyTorch3D mesh rasterizer.

    Args:
        cameras: PyTorch3D cameras used for world-to-screen projection.
        resolution: Target image size `(height, width)`.
        blur_radius: Rasterization blur radius in normalized coordinates.
        faces_per_pixel: Number of faces retained at each pixel.
        clip_barycentric_coords: Whether PyTorch3D clips barycentric coordinates.

    Returns:
        MeshRasterizer configured for the requested camera and raster.
    """

    def _validate_inputs() -> None:
        assert isinstance(cameras, CamerasBase), (
            "Expected `cameras` to be a PyTorch3D CamerasBase. " f"{type(cameras)=}"
        )
        assert isinstance(resolution, tuple) and len(resolution) == 2, (
            "Expected `resolution` to be a length-2 tuple. " f"{resolution=}"
        )
        for axis_name, value in zip(("height", "width"), resolution, strict=True):
            assert isinstance(value, int) and not isinstance(value, bool), (
                "Expected each render resolution value to be an int. "
                f"{axis_name=} {type(value)=} {resolution=}"
            )
            assert value > 0, (
                "Expected render resolution values to be positive integers. "
                f"{axis_name=} {value=} {resolution=}"
            )
        assert isinstance(blur_radius, float), (
            "Expected `blur_radius` to be a float. " f"{type(blur_radius)=}"
        )
        assert (
            isinstance(faces_per_pixel, int)
            and not isinstance(faces_per_pixel, bool)
            and faces_per_pixel > 0
        ), ("Expected `faces_per_pixel` to be a positive int. " f"{faces_per_pixel=}")
        assert isinstance(clip_barycentric_coords, bool), (
            "Expected `clip_barycentric_coords` to be a bool. "
            f"{type(clip_barycentric_coords)=}"
        )

    _validate_inputs()

    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=False,
        clip_barycentric_coords=clip_barycentric_coords,
        bin_size=0,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


def _build_phong_shader(
    cameras: CamerasBase,
    device: torch.device,
    background_color: Tuple[int, int, int],
) -> SoftPhongShader:
    """Build a flat-ambient PyTorch3D Phong shader.

    Args:
        cameras: PyTorch3D cameras used by the shader.
        device: Torch device where shader tensors should be allocated.
        background_color: RGB background tuple with integer channels in `[0, 255]`.

    Returns:
        SoftPhongShader with flat ambient lighting and the requested background.
    """

    def _validate_inputs() -> None:
        assert isinstance(cameras, CamerasBase), (
            "Expected `cameras` to be a PyTorch3D CamerasBase. " f"{type(cameras)=}"
        )
        assert isinstance(device, torch.device), (
            "Expected `device` to be a torch.device. " f"{type(device)=}"
        )
        assert isinstance(background_color, tuple) and len(background_color) == 3, (
            "Expected `background_color` to be an RGB tuple. " f"{background_color=}"
        )
        for channel_name, channel in zip(
            ("red", "green", "blue"), background_color, strict=True
        ):
            assert isinstance(channel, int) and not isinstance(channel, bool), (
                "Expected background channels to be integers. "
                f"{channel_name=} {type(channel)=} {background_color=}"
            )
            assert 0 <= channel <= 255, (
                "Expected background channels to be in [0, 255]. "
                f"{channel_name=} {channel=} {background_color=}"
            )

    _validate_inputs()

    background = tuple(float(channel) / 255.0 for channel in background_color)

    lights = PointLights(
        device=device,
        location=torch.zeros(1, 3, device=device),
        ambient_color=torch.ones(1, 3, device=device),
        diffuse_color=torch.zeros(1, 3, device=device),
        specular_color=torch.zeros(1, 3, device=device),
    )

    blend_params = BlendParams(background_color=background)

    shader = SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params,
    )
    return shader


def _build_silhouette_shader(
    blend_sigma: float,
    blend_gamma: float,
) -> SoftSilhouetteShader:
    """Build PyTorch3D's silhouette shader for the given blend parameters.

    Args:
        blend_sigma: Soft-rasterization sigma of the silhouette blend.
        blend_gamma: Soft-rasterization gamma of the silhouette blend.

    Returns:
        A `SoftSilhouetteShader` carrying those blend parameters.
    """

    def _validate_inputs() -> None:
        assert isinstance(blend_sigma, float) and blend_sigma > 0.0, (
            "Expected `blend_sigma` to be a positive float. " f"{blend_sigma=}"
        )
        assert isinstance(blend_gamma, float) and blend_gamma > 0.0, (
            "Expected `blend_gamma` to be a positive float. " f"{blend_gamma=}"
        )

    _validate_inputs()

    blend_params = BlendParams(sigma=blend_sigma, gamma=blend_gamma)
    shader = SoftSilhouetteShader(blend_params=blend_params)
    return shader


@torch.no_grad()
def render_rgb_from_mesh(
    mesh: Mesh,
    camera: Camera,
    resolution: Optional[Tuple[int, int]] = None,
    background: Tuple[int, int, int] = (0, 0, 0),
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render an RGB image from a triangle mesh using PyTorch3D.

    Args:
        mesh: Repo `Mesh` object to render.
        camera: Camera containing intrinsics/extrinsics/convention.
        resolution: Optional target image size (height, width); `None` uses the camera's own resolution.
        background: Background RGB color as integer tuple in [0, 255].
        return_mask: If True, also return valid pixel mask (default: False).

    Returns:
        If return_mask is False:
            RGB image tensor of shape [3, H, W] in range [0, 1].
        If return_mask is True:
            Tuple of (RGB image tensor, valid mask tensor of shape [H, W]).
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a Mesh instance. " f"{type(mesh)=}"
        )
        assert isinstance(camera, Camera), (
            "Expected `camera` to be a Camera instance. " f"{type(camera)=}"
        )
        assert resolution is None or (
            isinstance(resolution, tuple) and len(resolution) == 2
        ), f"Expected `resolution` to be `None` or a length-2 tuple. {resolution=}"
        if resolution is not None:
            for axis_name, value in zip(("height", "width"), resolution, strict=True):
                assert isinstance(value, (int, torch.Tensor)) and not isinstance(
                    value, bool
                ), (
                    "Expected each render resolution value to be an int or scalar tensor. "
                    f"{axis_name=} {type(value)=} {resolution=}"
                )
                if isinstance(value, torch.Tensor):
                    assert value.numel() == 1, (
                        "Expected tensor render resolution values to contain one value. "
                        f"{axis_name=} {value.shape=}"
                    )
                    assert value.dtype != torch.bool, (
                        "Expected tensor render resolution values not to be bool. "
                        f"{axis_name=} {value.dtype=} {resolution=}"
                    )
                    assert not value.detach().is_floating_point() or torch.equal(
                        value.detach(), torch.round(value.detach())
                    ), (
                        "Expected tensor render resolution values to be integer-valued. "
                        f"{axis_name=} {value=}"
                    )
                    assert bool((value.detach() > 0).cpu().item()), (
                        "Expected tensor render resolution values to be positive. "
                        f"{axis_name=} {value=}"
                    )
                else:
                    assert value > 0, (
                        "Expected render resolution values to be positive integers. "
                        f"{axis_name=} {value=} {resolution=}"
                    )
        assert isinstance(background, tuple) and len(background) == 3, (
            "Expected `background` to be an RGB tuple. " f"{background=}"
        )
        for channel_name, channel in zip(
            ("red", "green", "blue"), background, strict=True
        ):
            assert isinstance(channel, int) and not isinstance(channel, bool), (
                "Expected background channels to be integers. "
                f"{channel_name=} {type(channel)=} {background=}"
            )
            assert 0 <= channel <= 255, (
                "Expected background channels to be in [0, 255]. "
                f"{channel_name=} {channel=} {background=}"
            )
        assert isinstance(return_mask, bool), (
            "Expected `return_mask` to be a bool. " f"{type(return_mask)=}"
        )

    _validate_inputs()

    def _normalize_inputs(
        resolution: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        if resolution is None:
            resolution = camera.intrinsics.resolution

        resolved = []
        for value in resolution:
            if isinstance(value, torch.Tensor):
                resolved.append(int(value.detach().cpu().item()))
            else:
                resolved.append(value)
        return resolved[0], resolved[1]

    resolution = _normalize_inputs(resolution=resolution)

    assert torch.cuda.is_available(), (
        "Expected CUDA to be available for mesh rendering. "
        f"{torch.cuda.is_available()=}"
    )
    device = torch.device("cuda")
    meshes = mesh_to_pytorch3d(
        mesh=mesh,
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(meshes, Meshes), (
        "Expected mesh conversion to produce PyTorch3D Meshes. " f"{type(meshes)=}"
    )

    cameras = _prepare_cameras(
        camera=camera,
        resolution=resolution,
        device=device,
    )

    rasterizer = _build_rasterizer(
        cameras=cameras,
        resolution=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
        clip_barycentric_coords=False,
    )
    shader = _build_phong_shader(
        cameras=cameras,
        device=device,
        background_color=background,
    )

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

        render_result = (rgb, valid_mask)
        return render_result
    return rgb


def render_soft_mask_from_mesh(
    mesh: Mesh,
    camera: Camera,
    blend_sigma: float,
    blend_gamma: float,
    faces_per_pixel: int,
    coverage_threshold: float,
    resolution: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Render a continuous coverage mask from a triangle mesh using PyTorch3D.

    Args:
        mesh: Repo `Mesh` object to render; its world-space vertices are used as they are.
        camera: Camera containing intrinsics/extrinsics/convention.
        blend_sigma: Soft-rasterization sigma of the silhouette blend.
        blend_gamma: Soft-rasterization gamma of the silhouette blend.
        faces_per_pixel: Number of faces each pixel blends over.
        coverage_threshold: Coverage value at which the blur past a face's own edge ends.
        resolution: Optional target image size (height, width); `None` uses the camera's own resolution.

    Returns:
        Coverage mask tensor of shape [H, W], float32 in range [0, 1].
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), f"Expected `mesh` to be a `Mesh`. {type(mesh)=}"
        assert isinstance(
            camera, Camera
        ), f"Expected `camera` to be a `Camera`. {type(camera)=}"
        assert isinstance(blend_sigma, float) and blend_sigma > 0.0, (
            "Expected `blend_sigma` to be a positive float. " f"{blend_sigma=}"
        )
        assert isinstance(blend_gamma, float) and blend_gamma > 0.0, (
            "Expected `blend_gamma` to be a positive float. " f"{blend_gamma=}"
        )
        assert (
            isinstance(faces_per_pixel, int)
            and not isinstance(faces_per_pixel, bool)
            and faces_per_pixel > 0
        ), ("Expected `faces_per_pixel` to be a positive int. " f"{faces_per_pixel=}")
        assert (
            isinstance(coverage_threshold, float) and 0.0 < coverage_threshold < 1.0
        ), (
            "Expected `coverage_threshold` to be a float in (0, 1). "
            f"{coverage_threshold=}"
        )
        assert resolution is None or (
            isinstance(resolution, tuple) and len(resolution) == 2
        ), f"Expected `resolution` to be `None` or a length-2 tuple. {resolution=}"
        if resolution is not None:
            for axis_name, value in zip(("height", "width"), resolution, strict=True):
                assert isinstance(value, (int, torch.Tensor)) and not isinstance(
                    value, bool
                ), (
                    "Expected each render resolution value to be an int or scalar tensor. "
                    f"{axis_name=} {type(value)=} {resolution=}"
                )
                if isinstance(value, torch.Tensor):
                    assert value.numel() == 1, (
                        "Expected tensor render resolution values to contain one value. "
                        f"{axis_name=} {value.shape=}"
                    )
                    assert value.dtype != torch.bool, (
                        "Expected tensor render resolution values not to be bool. "
                        f"{axis_name=} {value.dtype=} {resolution=}"
                    )
                    assert not value.detach().is_floating_point() or torch.equal(
                        value.detach(), torch.round(value.detach())
                    ), (
                        "Expected tensor render resolution values to be integer-valued. "
                        f"{axis_name=} {value=}"
                    )
                    assert bool((value.detach() > 0).cpu().item()), (
                        "Expected tensor render resolution values to be positive. "
                        f"{axis_name=} {value=}"
                    )
                else:
                    assert value > 0, (
                        "Expected render resolution values to be positive integers. "
                        f"{axis_name=} {value=} {resolution=}"
                    )

    _validate_inputs()

    def _normalize_inputs(
        resolution: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        if resolution is None:
            resolution = camera.intrinsics.resolution

        resolved = []
        for value in resolution:
            if isinstance(value, torch.Tensor):
                resolved.append(int(value.detach().cpu().item()))
            else:
                resolved.append(value)
        return resolved[0], resolved[1]

    resolution = _normalize_inputs(resolution=resolution)

    assert torch.cuda.is_available(), (
        "Expected CUDA to be available for mesh rendering. "
        f"{torch.cuda.is_available()=}"
    )
    device = torch.device("cuda")
    meshes = mesh_to_pytorch3d(
        mesh=mesh,
        device=device,
        dtype=torch.float32,
    )
    assert isinstance(meshes, Meshes), (
        "Expected mesh conversion to produce PyTorch3D Meshes. " f"{type(meshes)=}"
    )

    cameras = _prepare_cameras(
        camera=camera,
        resolution=resolution,
        device=device,
    )

    blur_radius = log(1 / coverage_threshold - 1) * blend_sigma
    rasterizer = _build_rasterizer(
        cameras=cameras,
        resolution=resolution,
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=True,
    )
    shader = _build_silhouette_shader(
        blend_sigma=blend_sigma,
        blend_gamma=blend_gamma,
    )
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    images = renderer(meshes)

    mask = images[0, :, :, 3]
    return mask
