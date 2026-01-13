import math
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from data.structures.three_d.camera.camera import Camera
from models.three_d.original_3dgs.model import GaussianModel
from models.three_d.original_3dgs.render.core import (
    _prepare_scaling_modifier,
    eval_sh,
    focal2fov,
    GaussianSplattingCamera,
)


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    separate_sh: bool = False,
    override_color=None,
    use_trained_exp: bool = False,
    max_splats: int = None,
    splat_fraction: float = None,
    distance_lod: Dict[str, float | int] | None = None,
    splat_pool_size: int = None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    device = pc.get_xyz.device
    total_splats = pc.get_xyz.shape[0]

    target_count = total_splats
    if splat_fraction is not None:
        target_count = max(1, int(total_splats * splat_fraction))
    if max_splats is not None:
        target_count = min(target_count, max_splats)
    target_count = max(1, min(target_count, total_splats))
    initial_target_count = target_count

    pool_size = splat_pool_size if splat_pool_size is not None else target_count
    pool_size = max(target_count, min(pool_size, total_splats))
    pool_indices = torch.arange(pool_size, device=device)

    base_indices = pool_indices[:target_count]
    render_positions = torch.arange(base_indices.shape[0], device=device)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points_full = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device
        )
        + 0
    )
    try:
        screenspace_points_full.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Optional distance-based LOD: cull splats before rasterization, then fade opacities to avoid pop-in.
    distance_lod_cfg = None
    if distance_lod and distance_lod.get("enabled", False):
        dmin = distance_lod.get("dmin", 0.0)
        dmax = distance_lod.get("dmax", 0.0)
        fade_start_ratio = float(distance_lod.get("fade_start_ratio", 0.2))
        fade_start_ratio = max(0.0, min(1.0, fade_start_ratio))
        n_high_val = distance_lod.get("n_high")
        n_low_val = distance_lod.get("n_low")

        available = base_indices.shape[0]
        n_high = available if n_high_val is None else n_high_val
        n_low = n_high if n_low_val is None else n_low_val
        n_high = max(1, min(int(n_high), available))
        n_low = max(1, min(int(n_low), available))

        distance_span = max(dmax - dmin, 1e-6)
        distances_full = torch.linalg.norm(
            pc.get_xyz[base_indices] - viewpoint_camera.camera_center[None], dim=1
        )
        beta = torch.clamp((distances_full - dmin) / distance_span, 0.0, 1.0)
        n_cull = beta * n_low + (1.0 - beta) * n_high
        importance_indices = torch.arange(
            1, available + 1, device=device, dtype=distances_full.dtype
        )
        visibility_mask = importance_indices <= n_cull
        if not torch.any(visibility_mask):
            keep_idx = int(torch.argmin(distances_full))
            visibility_mask[keep_idx] = True

        render_positions = torch.nonzero(visibility_mask, as_tuple=False).flatten()
        base_indices = base_indices[visibility_mask]
        distance_lod_cfg = {
            "dmin": dmin,
            "distance_span": distance_span,
            "fade_start_ratio": fade_start_ratio,
            "enabled": True,
        }

    target_count = base_indices.shape[0]

    means3D = pc.get_xyz[base_indices]
    means2D = screenspace_points_full[base_indices]
    opacity = pc.get_opacity[base_indices]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)[base_indices]
    else:
        scales = pc.get_scaling[base_indices]
        rotations = pc.get_rotation[base_indices]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    dc = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )[base_indices]
            dir_pp = pc.get_xyz[base_indices] - viewpoint_camera.camera_center.repeat(
                target_count, 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = (
                    pc.get_features_dc[base_indices],
                    pc.get_features_rest[base_indices],
                )
            else:
                shs = pc.get_features[base_indices]
    else:
        colors_precomp = override_color

    if distance_lod_cfg and distance_lod_cfg.get("enabled", False):
        distances = torch.linalg.norm(
            means3D - viewpoint_camera.camera_center[None], dim=1
        )
        fade_start = (
            distance_lod_cfg["dmin"]
            + distance_lod_cfg["fade_start_ratio"] * distance_lod_cfg["distance_span"]
        )
        fade_span = max(
            distance_lod_cfg["distance_span"]
            * (1.0 - distance_lod_cfg["fade_start_ratio"]),
            1e-6,
        )
        fade_factor = 1.0 - torch.relu(distances - fade_start) / fade_span
        fade = torch.clamp(fade_factor, 0.0, 1.0).unsqueeze(1)
        opacity = opacity * fade

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rendered_image, radii_rendered, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_image, radii_rendered, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    radii = torch.zeros(initial_target_count, device=device, dtype=radii_rendered.dtype)
    radii[render_positions] = radii_rendered

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(
                2, 0, 1
            )
            + exposure[:3, 3, None, None]
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    visibility_filter = torch.nonzero(radii > 0, as_tuple=False).flatten()
    gaussians_used = visibility_filter.shape[0]
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points_full,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": depth_image,
        "gaussians_used": gaussians_used,
    }

    return out


def build_distance_lod_config(
    total_gaussians: int,
    dmin: float,
    dmax: float,
    n_high_fraction: float,
    n_low_fraction: float,
) -> Dict[str, float | int]:
    # Input validations
    assert isinstance(total_gaussians, int), f"{type(total_gaussians)=}"
    assert total_gaussians > 0, f"{total_gaussians=}"
    assert isinstance(dmin, (int, float)), f"{type(dmin)=}"
    assert isinstance(dmax, (int, float)), f"{type(dmax)=}"
    assert math.isfinite(dmin) and math.isfinite(dmax), f"{dmin=}, {dmax=}"
    assert dmax > dmin, f"Distance window must satisfy dmax>dmin (got {dmin}, {dmax})"
    assert isinstance(n_high_fraction, (int, float)), f"{type(n_high_fraction)=}"
    assert isinstance(n_low_fraction, (int, float)), f"{type(n_low_fraction)=}"
    assert math.isfinite(n_high_fraction) and math.isfinite(n_low_fraction)
    assert n_high_fraction > 0.0, f"{n_high_fraction=}"
    assert n_low_fraction > 0.0, f"{n_low_fraction=}"

    n_high = int(math.ceil(n_high_fraction * float(total_gaussians)))
    n_low = int(math.ceil(n_low_fraction * float(total_gaussians)))
    n_high = max(1, min(total_gaussians, n_high))
    n_low = max(1, min(total_gaussians, n_low))
    assert n_high >= n_low, f"n_high ({n_high}) must be >= n_low ({n_low})"

    return {
        "dmin": float(dmin),
        "dmax": float(dmax),
        "n_high": int(n_high),
        "n_low": int(n_low),
    }


def distance_lod_mask(
    positions: torch.Tensor,
    camera_center: torch.Tensor,
    config: Dict[str, float | int],
) -> torch.Tensor:
    # Input validations
    assert torch.is_tensor(positions), f"{type(positions)=}"
    assert positions.ndim == 2 and positions.shape[1] == 3, f"{positions.shape=}"
    assert torch.is_tensor(camera_center), f"{type(camera_center)=}"
    assert camera_center.shape == (3,), f"{camera_center.shape=}"
    assert isinstance(config, dict), f"{type(config)=}"
    for key in ("dmin", "dmax", "n_high", "n_low"):
        assert key in config, f"Distance LOD config missing '{key}'"

    dmin = float(config["dmin"])
    dmax = float(config["dmax"])
    n_high = int(config["n_high"])
    n_low = int(config["n_low"])

    assert math.isfinite(dmin) and math.isfinite(dmax), f"{dmin=}, {dmax=}"
    assert dmax > dmin, f"dmax must exceed dmin ({dmin}, {dmax})"
    assert n_high > 0 and n_low > 0, f"{n_high=}, {n_low=}"
    total = positions.shape[0]
    assert n_high <= total and n_low <= total, f"{n_high=}, {n_low=}, {total=}"

    distance_span = dmax - dmin
    distances = torch.linalg.norm(
        positions - camera_center.to(device=positions.device, dtype=positions.dtype),
        dim=1,
    )
    beta = torch.clamp((distances - dmin) / distance_span, 0.0, 1.0)
    n_cull = beta * float(n_low) + (1.0 - beta) * float(n_high)
    importance_indices = torch.arange(
        1, total + 1, device=positions.device, dtype=distances.dtype
    )
    mask = importance_indices <= n_cull
    assert mask.any(), "Distance LOD mask must retain at least one Gaussian"
    return mask


def distance_lod_mask_for_camera(
    positions: torch.Tensor,
    camera: Camera,
    resolution: tuple[int, int],
    device: torch.device,
    config: Dict[str, float | int],
) -> torch.Tensor:
    # Input validations
    assert torch.is_tensor(positions), f"{type(positions)=}"
    assert positions.ndim == 2 and positions.shape[1] == 3, f"{positions.shape=}"
    assert hasattr(camera, 'intrinsics'), "Camera must expose intrinsics"
    assert isinstance(resolution, tuple) and len(resolution) == 2, f"{resolution=}"
    assert isinstance(device, torch.device), f"{type(device)=}"

    viewpoint_camera = _build_viewpoint_camera(
        camera=camera, resolution=resolution, device=device
    )
    return distance_lod_mask(
        positions=positions,
        camera_center=viewpoint_camera.camera_center,
        config=config,
    )


def _build_viewpoint_camera(
    camera: Camera, resolution: Tuple[int, int], device: torch.device
) -> GaussianSplattingCamera:
    # Input validations
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert isinstance(resolution, tuple) and len(resolution) == 2, f"{resolution=}"
    assert all(isinstance(v, int) for v in resolution), f"{resolution=}"
    assert isinstance(device, torch.device), f"{type(device)=}"

    base_width = int(float(camera.cx) * 2.0)
    base_height = int(float(camera.cy) * 2.0)
    assert base_width > 0 and base_height > 0, f"{base_width=}, {base_height=}"

    fov_x = focal2fov(float(camera.fx), base_width)
    fov_y = focal2fov(float(camera.fy), base_height)

    camera_opencv = camera.to(device=device, convention="opencv")
    w2c = camera_opencv.w2c.detach().cpu().numpy()
    rotation = np.transpose(w2c[:3, :3])
    translation = w2c[:3, 3]

    dummy_image = Image.new("RGB", (resolution[1], resolution[0]), color=(0, 0, 0))

    return GaussianSplattingCamera(
        resolution=(resolution[1], resolution[0]),
        colmap_id=0,
        R=rotation,
        T=translation,
        FoVx=fov_x,
        FoVy=fov_y,
        depth_params=None,
        image=dummy_image,
        invdepthmap=None,
        image_name="render",
        uid=0,
        data_device=str(device),
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
    )


def render_rgb_from_milef2025learning(
    model: GaussianModel,
    camera: Camera,
    resolution: Tuple[int, int],
    distance_lod_enabled: bool = False,
    distance_dmin: float = 1.0,
    distance_dmax: float = 5.0,
    distance_n_high: int = 1,
    distance_n_low: int = 1,
    distance_fade_start_ratio: float = 0.2,
    background: Tuple[int, int, int] = (0, 0, 0),
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    # Input validations
    assert isinstance(model, GaussianModel), f"{type(model)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert isinstance(resolution, tuple) and len(resolution) == 2, f"{resolution=}"
    assert all(isinstance(v, int) for v in resolution), f"{resolution=}"
    assert isinstance(distance_lod_enabled, bool), f"{type(distance_lod_enabled)=}"
    assert isinstance(distance_dmin, (int, float)), f"{type(distance_dmin)=}"
    assert isinstance(distance_dmax, (int, float)), f"{type(distance_dmax)=}"
    assert (
        not distance_lod_enabled or distance_dmax > distance_dmin
    ), f"{distance_dmin=}, {distance_dmax=}"
    assert isinstance(distance_n_high, int), f"{type(distance_n_high)=}"
    assert isinstance(distance_n_low, int), f"{type(distance_n_low)=}"
    assert isinstance(
        distance_fade_start_ratio, (int, float)
    ), f"{type(distance_fade_start_ratio)=}"
    assert (
        not distance_lod_enabled or 0.0 <= float(distance_fade_start_ratio) < 1.0
    ), f"{distance_fade_start_ratio=}"
    assert isinstance(background, tuple) and len(background) == 3, f"{background=}"
    assert isinstance(device, torch.device), f"{type(device)=}"

    device = model.get_xyz.device
    scaling_modifier = _prepare_scaling_modifier(
        intrinsics=camera.intrinsics, resolution=resolution
    )
    viewpoint_camera = _build_viewpoint_camera(
        camera=camera, resolution=resolution, device=device
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    background_tensor = torch.tensor(background, dtype=torch.float32, device=device)
    distance_cfg = None
    if distance_lod_enabled:
        distance_cfg = {
            "enabled": True,
            "dmin": float(distance_dmin),
            "dmax": float(distance_dmax),
            "n_high": int(distance_n_high),
            "n_low": int(distance_n_low),
            "fade_start_ratio": float(distance_fade_start_ratio),
        }
    outputs = render(
        viewpoint_camera=viewpoint_camera,
        pc=model,
        pipe=pipeline,
        bg_color=background_tensor,
        scaling_modifier=scaling_modifier,
        distance_lod=distance_cfg,
    )
    assert "render" in outputs, "render output missing from Milef2025Learning renderer"
    return outputs["render"]
