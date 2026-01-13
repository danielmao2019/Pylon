import math
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from data.structures.three_d.camera.camera import Camera
from models.three_d.cheng2025clod.model import Cheng2025CLODGS
from models.three_d.original_3dgs.render.core import (
    _prepare_scaling_modifier,
    eval_sh,
    focal2fov,
    GaussianSplattingCamera,
)


def geom_transform_points(
    points: torch.Tensor, transf_matrix: torch.Tensor
) -> torch.Tensor:
    assert torch.is_tensor(points), f"{type(points)=}"
    assert torch.is_tensor(transf_matrix), f"{type(transf_matrix)=}"
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


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


def render(
    viewpoint_camera,
    pc: Cheng2025CLODGS,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    separate_sh: bool = False,
    override_color=None,
    use_trained_exp: bool = False,
    virtual_scale: float = 1.0,
    lod_threshold: float = 0.01,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
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

    means3D = pc.get_xyz
    means2D = screenspace_points
    base_opacity = pc.get_opacity

    # Distance-adaptive opacity for continuous LoD
    distances = torch.linalg.norm(
        means3D - viewpoint_camera.camera_center, dim=1, keepdim=True
    )
    # Use only in-frustum Gaussians to set the distance scale; fall back to all points if none are visible.
    ndc_means = geom_transform_points(means3D, viewpoint_camera.full_proj_transform)
    in_frustum = (
        (ndc_means[:, 2] > 0)
        & (ndc_means[:, 2] <= 1)
        & (torch.abs(ndc_means[:, 0]) <= 1)
        & (torch.abs(ndc_means[:, 1]) <= 1)
    )
    if torch.any(in_frustum):
        max_distance = torch.clamp(torch.max(distances[in_frustum]), min=1e-6)
    else:
        max_distance = torch.clamp(torch.max(distances), min=1e-6)
    normalized_distance = distances / max_distance
    sigma = pc.get_lod_sigma
    attenuation = torch.exp(
        -((normalized_distance * virtual_scale) ** 2) / (2.0 * sigma**2 + 1e-6)
    )
    attenuated_opacity = base_opacity * attenuation
    lod_mask = torch.sigmoid(attenuated_opacity - (lod_threshold * virtual_scale))
    opacity = attenuated_opacity * lod_mask

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dc = None
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
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
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(
                2, 0, 1
            )
            + exposure[:3, 3, None, None]
        )

    rendered_image = rendered_image.clamp(0, 1)
    lod_mask_flat = lod_mask.squeeze(-1)
    # Soft LoD visibility for differentiable usage; hard mask remains for stats/filters.
    lod_visibility = lod_mask_flat * (radii > 0).float()
    visible_mask = torch.logical_and(radii > 0, lod_mask_flat > 0.5)
    visibility_filter = visible_mask.nonzero()
    gaussians_used = visible_mask.sum()
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": depth_image,
        "gaussians_used": gaussians_used,
        "lod_mask": lod_mask_flat,
        "lod_visibility": lod_visibility,
        "visible_mask": visible_mask,
        "attenuated_opacity": attenuated_opacity,
    }

    return out


@torch.no_grad()
def render_rgb_from_cheng2025_clod(
    model: Cheng2025CLODGS,
    camera: Camera,
    resolution: Tuple[int, int],
    virtual_scale: float,
    lod_threshold: float,
    background: Tuple[int, int, int] = (0, 0, 0),
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    # Input validations
    assert isinstance(model, Cheng2025CLODGS), f"{type(model)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert isinstance(resolution, tuple) and len(resolution) == 2, f"{resolution=}"
    assert all(isinstance(v, int) for v in resolution), f"{resolution=}"
    assert isinstance(virtual_scale, (int, float)), f"{type(virtual_scale)=}"
    assert isinstance(lod_threshold, (int, float)), f"{type(lod_threshold)=}"
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

    outputs = render(
        viewpoint_camera=viewpoint_camera,
        pc=model,
        pipe=pipeline,
        bg_color=background_tensor,
        scaling_modifier=scaling_modifier,
        virtual_scale=virtual_scale,
        lod_threshold=lod_threshold,
    )
    assert "render" in outputs, "render output missing from Cheng2025CLOD renderer"
    return outputs["render"]
