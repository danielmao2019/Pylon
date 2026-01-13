import copy
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import gsplat
import numpy as np
import torch
import torch.nn as nn
from gsplat.cuda._wrapper import fully_fused_projection

from data.structures.three_d.camera.camera import Camera
from models.three_d.octree_gs.loader import OctreeGS_3DGS


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


class OctreeGSCamera(nn.Module):

    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        image_name,
        resolution_scale,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(OctreeGSCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution_scale = resolution_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

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


def filter_octree_gs_by_levels(
    model: OctreeGS_3DGS,
    viewpoint_camera: OctreeGSCamera,
    levels: List[int],
    resolution_scale: float = 1.0,
    return_info: bool = False,
) -> Union[OctreeGS_3DGS, Tuple[OctreeGS_3DGS, Dict[str, Any]]]:
    """Filter Octree GS model to only include gaussians at specified levels.

    This function creates a filtered version of the model containing only the gaussians
    that would be rendered at the specified levels given the current camera position.

    Args:
        model: Original OctreeGS_3DGS instance
        viewpoint_camera: Camera object containing camera_center
        levels: List of levels to include (each 0 to model.levels-1)
        resolution_scale: Resolution scale factor
        return_info: If True, also return gaussian counts per level

    Returns:
        If return_info is False:
            A new OctreeGS_3DGS with only the gaussians at the specified levels
        If return_info is True:
            Tuple of (filtered_model, info_dict) where:
                - filtered_model: OctreeGS_3DGS with only selected levels
                - info_dict: Dict containing 'gaussian_counts_per_level' and 'total_gaussians'
    """
    # Validate levels (should be validated by caller, but double-check)
    assert isinstance(levels, list), f"levels must be list, got {type(levels)}"
    assert len(levels) > 0, f"levels must be non-empty, got empty list"
    for level in levels:
        assert isinstance(level, int), f"each level must be int, got {type(level)}"
        assert level >= 0, f"each level must be non-negative, got {level}"
        assert (
            level < model.levels
        ), f"level {level} exceeds max available level {model.levels - 1}"

    # Compute which gaussians belong to the target levels
    dist = (
        torch.sqrt(
            torch.sum((model.get_anchor - viewpoint_camera.camera_center) ** 2, dim=1)
        )
        * resolution_scale
    )
    pred_level = (
        torch.log2(model.standard_dist / dist) / math.log2(model.fork)
        + model._extra_level
    )

    # Map to integer levels
    int_level = model.map_to_int_level(pred_level, model.levels - 1)

    # Count gaussians per level if requested
    if return_info:
        gaussian_counts_per_level = {}
        for level in range(model.levels):
            count = (int_level == level).sum().item()
            gaussian_counts_per_level[level] = count
        total_gaussians = int_level.shape[0]

    # Create mask for gaussians at any of the target levels
    mask = torch.zeros(int_level.shape[0], dtype=torch.bool, device=int_level.device)
    for level in levels:
        mask |= int_level == level

    # Make a shallow copy so we can overwrite masked tensors without touching the original model.
    filtered_model = copy.copy(model)

    filtered_model._anchor = nn.Parameter(filtered_model._anchor.detach()[mask].clone())

    filtered_model._level = filtered_model._level.detach()[mask].clone()

    filtered_model._extra_level = filtered_model._extra_level.detach()[mask].clone()

    filtered_model._features_dc = nn.Parameter(
        filtered_model._features_dc.detach()[mask].clone()
    )

    filtered_model._features_rest = nn.Parameter(
        filtered_model._features_rest.detach()[mask].clone()
    )

    filtered_model._opacity = nn.Parameter(
        filtered_model._opacity.detach()[mask].clone()
    )

    filtered_model._offset = nn.Parameter(filtered_model._offset.detach()[mask].clone())

    filtered_model._scaling = nn.Parameter(
        filtered_model._scaling.detach()[mask].clone()
    )

    filtered_model._rotation = nn.Parameter(
        filtered_model._rotation.detach()[mask].clone()
    )

    # Reset the anchor mask to include all remaining gaussians
    filtered_model._anchor_mask = torch.ones(
        filtered_model._anchor.shape[0],
        dtype=torch.bool,
        device=filtered_model._anchor.device,
    )

    if return_info:
        info_dict = {
            'gaussian_counts_per_level': gaussian_counts_per_level,
            'total_gaussians': total_gaussians,
        }
        return filtered_model, info_dict
    else:
        return filtered_model


def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)

    Ks = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )[None]
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None]

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        None,  # covars,
        quats,
        scales,
        viewmats,
        Ks,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=False,
    )

    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    radii, means2d, depths, conics, compensations = proj_results
    camera_ids, gaussian_ids = None, None

    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0

    return visible_mask


def render(viewpoint_camera, pc, pipe, bg_color, iteration, render_mode, ape_code=-1):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    pc.set_anchor_mask(
        viewpoint_camera.camera_center, iteration, viewpoint_camera.resolution_scale
    )
    # Anchor-mask percentage needs to be captured immediately after updating the mask
    total_anchors = int(pc._anchor_mask.numel())
    active_anchors = int(pc._anchor_mask.sum().item()) if total_anchors > 0 else 0
    anchor_mask_percentage = (
        100.0 * active_anchors / total_anchors if total_anchors > 0 else 0.0
    )
    visible_mask = prefilter_voxel(viewpoint_camera, pc, pipe, bg_color)
    xyz, color, opacity, scaling, rot, sh_degree, selection_mask = (
        pc.generate_neural_gaussians(viewpoint_camera, visible_mask, ape_code)
    )

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
    render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,  # [N, 3]
        quats=rot,  # [N, 4]
        scales=scaling,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=color,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )

    # [1, H, W, 3] -> [3, H, W]
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)  # [N,]
    try:
        info["means2d"].retain_grad()  # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
        "anchor_mask_percentage": anchor_mask_percentage,
        "anchor_mask_active": active_anchors,
        "anchor_mask_total": total_anchors,
    }

    return return_dict


@torch.no_grad()
def render_rgb_from_octree_gs(
    model: OctreeGS_3DGS,
    camera: Camera,
    resolution: Optional[Tuple[int, int]] = None,
    background: Tuple[int, int, int] = (0, 0, 0),
    levels: Optional[List[int]] = None,
    return_info: bool = False,
    device: torch.device = torch.device('cuda'),
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Render RGB image from Octree Gaussian Splatting model.

    Args:
        model: OctreeGS_3DGS instance
        camera: Camera instance containing intrinsics/extrinsics/convention
        resolution: Optional (height, width) tuple for output resolution
        background: RGB background color tuple
        levels: Optional list of levels to render. If None, uses dynamic level based on camera distance.
                If provided, renders only Gaussians at the specified levels (each 0 to model.levels-1).
                Can be empty list to render only background.
        return_info: If True, return additional information including gaussian counts per level

    Returns:
        If return_info is False:
            Rendered RGB tensor of shape (3, height, width)
        If return_info is True:
            Tuple of (RGB tensor, info_dict) where info_dict contains:
                - 'gaussian_counts_per_level': Dict[int, int] mapping level to gaussian count
                - 'total_gaussians': int total number of gaussians
    """
    assert isinstance(model, OctreeGS_3DGS)
    assert isinstance(camera, Camera), f"{type(camera)=}"

    # Validate levels parameter if provided
    if levels is not None:
        assert isinstance(levels, list), f"levels must be list, got {type(levels)}"
        # If levels is empty, we'll return a background image later
        for level in levels:
            assert isinstance(level, int), f"each level must be int, got {type(level)}"
            assert level >= 0, f"each level must be non-negative, got {level}"
            assert (
                level < model.levels
            ), f"level {level} exceeds max available level {model.levels - 1}"
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
    assert (
        isinstance(background, tuple) and len(background) == 3
    ), "background must be an RGB tuple"
    assert all(
        isinstance(v, int) for v in background
    ), "background entries must be integers"

    # Enforce CUDA usage for both provided device and model device
    assert device.type == 'cuda', f"3DGS rendering requires CUDA, got '{device}'"
    assert (
        model.get_anchor.device.type == 'cuda'
    ), f"Model must reside on CUDA, got '{model.get_anchor.device}'"

    base_width = int(round(camera.cx * 2.0))
    base_height = int(round(camera.cy * 2.0))
    if base_width <= 0 or base_height <= 0:
        raise ValueError(
            "Unable to infer image resolution from intrinsics; provide explicit resolution"
        )

    if resolution is None:
        target_height = base_height
        target_width = base_width

    scale_x = target_width / base_width
    scale_y = target_height / base_height
    if not math.isclose(scale_x, scale_y, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            "Non-uniform scaling not supported for 3DGS rendering: "
            f"scale_x={scale_x}, scale_y={scale_y}"
        )
    resolution_scale = (scale_x + scale_y) / 2

    background_tensor = torch.tensor(background, dtype=torch.float32, device=device)

    # If levels is specified and empty, return background image
    if levels is not None and len(levels) == 0:
        # Create a background image with the specified background color
        # Shape: (3, height, width), normalized to [0, 1] range
        background_image = (
            background_tensor.view(3, 1, 1).expand(3, target_height, target_width)
            / 255.0
        )
        return background_image

    camera = camera.to(device=device, convention="opencv")
    w2c = camera.w2c.detach().cpu().numpy()
    rotation = np.transpose(w2c[:3, :3])
    translation = w2c[:3, 3]

    fov_x = focal2fov(camera.fx, base_width)
    fov_y = focal2fov(camera.fy, base_height)

    dummy_image = torch.zeros(
        size=(3, target_height, target_width), dtype=torch.float32, device=device
    )

    octree_gs_camera = OctreeGSCamera(
        R=rotation,
        T=translation,
        FoVx=fov_x,
        FoVy=fov_y,
        resolution_scale=resolution_scale,
        image=dummy_image,
        image_name="render",
        uid=0,
        colmap_id=0,
        data_device=str(device),
    )

    pipeline = SimpleNamespace(
        compute_cov3D_python=False,
        debug=False,
    )

    # Handle levels filtering and info collection
    info_dict = None
    if levels is not None:
        # Filter the model to only include gaussians at the specified levels
        filter_result = filter_octree_gs_by_levels(
            model=model,
            viewpoint_camera=octree_gs_camera,
            levels=levels,
            resolution_scale=resolution_scale,
            return_info=return_info,
        )

        if return_info:
            model_to_render, info_dict = filter_result
        else:
            model_to_render = filter_result
    else:
        # Use the original model for normal rendering
        model_to_render = model

        # If return_info=True but no filtering, still need to compute counts
        if return_info:
            dist = (
                torch.sqrt(
                    torch.sum((model.get_anchor - octree_gs_camera.camera_center) ** 2, dim=1)
                )
                * resolution_scale
            )
            pred_level = (
                torch.log2(model.standard_dist / dist) / math.log2(model.fork)
                + model._extra_level
            )
            int_level = model.map_to_int_level(pred_level, model.levels - 1)

            gaussian_counts_per_level = {}
            for level in range(model.levels):
                count = (int_level == level).sum().item()
                gaussian_counts_per_level[level] = count

            total_gaussians = int_level.shape[0]

            info_dict = {
                'gaussian_counts_per_level': gaussian_counts_per_level,
                'total_gaussians': total_gaussians,
            }

    # Render using the (potentially filtered) model
    outputs = render(
        viewpoint_camera=octree_gs_camera,
        pc=model_to_render,
        pipe=pipeline,
        bg_color=background_tensor,
        iteration=30000,
        render_mode='RGB',
    )
    rgb = outputs["render"]
    expected_shape = (3, target_height, target_width)
    assert (
        rgb.shape == expected_shape
    ), f"Unexpected RGB output shape {tuple(rgb.shape)}; expected {expected_shape}"

    if return_info:
        assert (
            info_dict is not None
        ), "info_dict must be populated when return_info=True"
        assert 'anchor_mask_percentage' in outputs
        anchor_mask_percentage = outputs['anchor_mask_percentage']
        assert isinstance(
            anchor_mask_percentage, float
        ), f"anchor_mask_percentage must be float, got {type(anchor_mask_percentage)=}"

        assert 'anchor_mask_active' in outputs
        anchor_mask_active = outputs['anchor_mask_active']
        assert isinstance(
            anchor_mask_active, int
        ), f"anchor_mask_active must be int, got {type(anchor_mask_active)=}"

        assert 'anchor_mask_total' in outputs
        anchor_mask_total = outputs['anchor_mask_total']
        assert isinstance(
            anchor_mask_total, int
        ), f"anchor_mask_total must be int, got {type(anchor_mask_total)=}"

        info_dict.update(
            {
                'anchor_mask_percentage': anchor_mask_percentage,
                'anchor_mask_active': anchor_mask_active,
                'anchor_mask_total': anchor_mask_total,
            }
        )
        return rgb, info_dict

    return rgb
