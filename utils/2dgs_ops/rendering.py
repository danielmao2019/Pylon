"""Rendering helpers for 2D Gaussian Splatting models.

This adapter mirrors the original 2DGS rendering entry by constructing a
MiniCam view object (matching scene/cameras.py:MiniCam) and invoking the
renderer exposed by the loaded model. It does not import from the 2DGS repo.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import numpy as np
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import importlib

GaussianModel = importlib.import_module('utils.io.2dgs').GaussianModel
from torch import nn

from utils.input_checks.check_camera import (
    check_camera_extrinsics,
    check_camera_intrinsics,
)
from utils.three_d.camera.conventions import apply_coordinate_transform


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = (
        torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
        .float()
        .cuda()
        .T
    )
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device='cuda').float(),
        torch.arange(H, device='cuda').float(),
        indexing='xy',
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
    view: view camera
    depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(
            0.0, 1.0
        )  # move to device at dataloader to reduce VRAM requirement
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # do we need this?
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )

                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
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
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = (
            torch.tensor(
                [
                    [W / 2, 0, 0, (W - 1) / 2],
                    [0, H / 2, 0, (H - 1) / 2],
                    [0, 0, far - near, near],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .cuda()
            .T
        )
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]])
            .permute(0, 2, 1)
            .reshape(-1, 9)
        )  # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
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
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0)
        @ (viewpoint_camera.world_view_transform[:3, :3].T)
    ).permute(2, 0, 1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = (
        render_depth_expected * (1 - pipe.depth_ratio)
        + (pipe.depth_ratio) * render_depth_median
    )

    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update(
        {
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        }
    )

    return rets


@torch.no_grad()
def render_rgb_from_2dgs(
    model: Any,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Optional[Tuple[int, int]] = None,
    convention: str = "opengl",
    background: Tuple[int, int, int] = (0, 0, 0),
    compute_cov3D_python: bool = False,
    convert_SHs_python: bool = False,
    depth_ratio: float = 0.0,
    debug: bool = False,
) -> torch.Tensor:
    """Convenience wrapper returning only the RGB image.

    Args:
        model: Pipeline wrapper exposing the 2DGS model.
        camera_intrinsics: Camera intrinsics matrix.
        camera_extrinsics: Camera-to-world transform.
        resolution: Optional (H, W) output size. If ``None``, derives from intrinsics.
        convention: Source convention for the extrinsics. Default: ``"opengl"``.
    """
    # input validations
    assert (
        model is not None
    ), "A 2DGS model wrapper or GaussianModel instance is required"
    assert hasattr(
        model, "model"
    ), "render_rgb_from_2dgs expects output of load_2dgs_model"
    check_camera_intrinsics(intrinsics)
    check_camera_extrinsics(extrinsics)
    if resolution is not None:
        assert isinstance(
            resolution, tuple
        ), "resolution must be a tuple of (height, width)"
        assert len(resolution) == 2, "resolution must be a tuple of (height, width)"
        target_height, target_width = resolution
        assert isinstance(target_height, int) and isinstance(
            target_width, int
        ), "resolution entries must be integers"
        assert (
            target_height > 0 and target_width > 0
        ), "resolution dimensions must be positive integers"
    assert convention in [
        "opengl",
        "standard",
        "opencv",
    ], "convention must be 'opengl', 'standard', or 'opencv'"
    assert (
        isinstance(background, tuple) and len(background) == 3
    ), "background must be an RGB tuple"
    assert all(
        isinstance(v, int) for v in background
    ), "background entries must be integers"

    # resolve input args
    gaussians = model.model
    base_width = int(round(float(intrinsics[0, 2]) * 2.0))
    base_height = int(round(float(intrinsics[1, 2]) * 2.0))
    if resolution is None:
        target_height = base_height
        target_width = base_width
    else:
        target_height, target_width = resolution
    device = gaussians.get_xyz.device
    if device.type != "cuda":
        raise ValueError(f"2DGS rendering requires a CUDA device, received '{device}'")
    background = torch.tensor(background, dtype=torch.float32, device=device)
    pipeline = SimpleNamespace(
        convert_SHs_python=convert_SHs_python,
        compute_cov3D_python=compute_cov3D_python,
        depth_ratio=depth_ratio,
        debug=debug,
    )

    # prepare camera intrinsics
    fov_x = focal2fov(float(intrinsics[0, 0]), base_width)
    fov_y = focal2fov(float(intrinsics[1, 1]), base_height)

    # prepare camera extrinsics
    extrinsics = (
        apply_coordinate_transform(
            extrinsics=extrinsics.detach()
            .clone()
            .to(device=device, dtype=torch.float32),
            source_convention=convention,
            target_convention="opencv",
        )
        .detach()
        .cpu()
        .numpy()
    )

    rotation = extrinsics[:3, :3].astype(np.float32)
    w2c = np.linalg.inv(extrinsics)
    translation = w2c[:3, 3].astype(np.float32)

    # prepare dummy image
    dummy_image = torch.zeros(
        (3, target_height, target_width), dtype=torch.float32, device=device
    )

    # define camera
    camera = Camera(
        colmap_id=0,
        R=rotation,
        T=translation,
        FoVx=fov_x,
        FoVy=fov_y,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name="render",
        uid=0,
        data_device=str(device),
    )

    # define scale
    scale_x = target_width / base_width
    scale_y = target_height / base_height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            "Non-uniform scaling not supported for 2DGS rendering: "
            f"scale_x={scale_x}, scale_y={scale_y}"
        )

    outputs = render(
        viewpoint_camera=camera,
        pc=gaussians,
        pipe=pipeline,
        bg_color=background,
        scaling_modifier=scale_x,
    )

    rgb = outputs["render"]
    expected_shape = (3, target_height, target_width)
    assert (
        rgb.shape == expected_shape
    ), f"Unexpected RGB output shape {tuple(rgb.shape)}; expected {expected_shape}"
    return rgb
