import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.octree_gs.loader import OctreeGS_3DGS
from models.three_d.octree_gs.render import (
    render_density_from_octree_gs,
    render_rgb_from_octree_gs,
)
from models.three_d.octree_gs.render.rgb import OctreeGSCamera
from models.three_d.octree_gs.render.rgb import focal2fov


def render_display(
    scene_model: BaseSceneModel,
    camera: Camera,
    resolution: Tuple[int, int],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    debugger_enabled: bool,
    selected_levels_rgb: List[int],
    selected_levels_density: List[int],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    device: Optional[torch.device],
) -> Dict[str, Any]:
    # Input validation
    assert isinstance(scene_model, BaseSceneModel), f"{type(scene_model)=}"
    assert isinstance(dataset_name, str) and dataset_name, f"{dataset_name=}"
    assert isinstance(scene_name, str) and scene_name, f"{scene_name=}"
    assert isinstance(method_name, str) and method_name, f"{method_name=}"
    assert isinstance(debugger_enabled, bool), f"{type(debugger_enabled)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert resolution and isinstance(resolution, tuple) and len(resolution) == 2
    assert camera_name is None or isinstance(camera_name, str), f"{type(camera_name)=}"
    assert device is None or isinstance(device, torch.device), f"{type(device)=}"
    if display_cameras is not None:
        assert isinstance(display_cameras, list), f"{type(display_cameras)=}"
        assert all(isinstance(cam, Camera) for cam in display_cameras)

    resolved_device = device if device is not None else scene_model.device
    assert isinstance(resolved_device, torch.device), f"{type(resolved_device)=}"
    assert (
        resolved_device.type == 'cuda'
    ), f"Expected CUDA device, got {resolved_device.type}"

    camera = camera.to(resolved_device)
    model = scene_model.model
    assert isinstance(
        model, OctreeGS_3DGS
    ), f"Expected OctreeGS_3DGS for octree_gs, got {type(model)}"

    total_levels = int(model.levels)
    full_levels = list(range(total_levels))
    cache_allowed = (
        camera_name is not None
        and not debugger_enabled
        and selected_levels_rgb == full_levels
    )
    if not debugger_enabled:
        selected_levels_rgb = full_levels
        selected_levels_density = full_levels
    if not cache_allowed:
        camera_name = None

    if debugger_enabled:
        return _render_display_debugger(
            scene_model=scene_model,
            model=model,
            camera=camera,
            resolution=resolution,
            dataset_name=dataset_name,
            scene_name=scene_name,
            method_name=method_name,
            levels_rgb=selected_levels_rgb,
            levels_density=selected_levels_density,
            camera_name=camera_name,
            display_cameras=display_cameras,
            title=title,
            total_levels=total_levels,
        )

    return _render_display_main(
        scene_model=scene_model,
        model=model,
        camera=camera,
        resolution=resolution,
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        levels_rgb=selected_levels_rgb,
        camera_name=camera_name,
        display_cameras=display_cameras,
        title=title,
    )


def _compute_debugger_info(
    model: OctreeGS_3DGS,
    camera: Camera,
    resolution: Tuple[int, int],
    total_levels: int,
) -> Dict[str, Any]:
    device = model.get_anchor.device
    assert device.type == 'cuda', f"Expected CUDA device, got {device}"
    camera = camera.to(device)

    base_width = int(round(camera.cx * 2.0))
    base_height = int(round(camera.cy * 2.0))
    assert (
        base_width > 0 and base_height > 0
    ), f"Invalid base resolution: {base_height=} {base_width=}"

    target_height, target_width = resolution
    assert isinstance(target_height, int) and isinstance(
        target_width, int
    ), f"{type(target_height)=}, {type(target_width)=}"
    assert (
        target_height > 0 and target_width > 0
    ), f"Resolution must be positive, got {target_height=} {target_width=}"

    scale_x = target_width / base_width
    scale_y = target_height / base_height
    assert math.isclose(
        scale_x, scale_y, rel_tol=1e-4, abs_tol=1e-4
    ), f"Non-uniform scaling: {scale_x=} {scale_y=}"
    resolution_scale = (scale_x + scale_y) / 2

    camera = camera.to(device=device, convention="opencv")
    w2c = camera.w2c.detach().cpu().numpy()
    rotation = np.transpose(w2c[:3, :3])
    translation = w2c[:3, 3]

    fov_x = focal2fov(camera.fx, base_width)
    fov_y = focal2fov(camera.fy, base_height)

    dummy_image = torch.zeros(
        size=(3, target_height, target_width), dtype=torch.float32, device=device
    )
    oct_camera = OctreeGSCamera(
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

    dist = (
        torch.sqrt(torch.sum((model.get_anchor - oct_camera.camera_center) ** 2, dim=1))
        * resolution_scale
    )
    assert torch.all(dist > 0), "All anchors must be at positive distance"
    pred_level = (
        torch.log2(model.standard_dist / dist) / math.log2(model.fork)
        + model._extra_level
    )
    int_level = model.map_to_int_level(pred_level, model.levels - 1)

    gaussian_counts_per_level: Dict[int, int] = {}
    for level in range(total_levels):
        gaussian_counts_per_level[level] = int((int_level == level).sum().item())
    total_gaussians = int(int_level.shape[0])
    assert total_gaussians > 0, "total_gaussians must be positive"

    model.set_anchor_mask(
        oct_camera.camera_center, iteration=30000, resolution_scale=resolution_scale
    )
    anchor_mask_total = int(model._anchor_mask.numel())
    assert anchor_mask_total > 0, "anchor_mask_total must be positive"
    anchor_mask_active = int(model._anchor_mask.sum().item())
    anchor_mask_percentage = 100.0 * anchor_mask_active / anchor_mask_total

    info = {
        'gaussian_counts_per_level': gaussian_counts_per_level,
        'total_gaussians': total_gaussians,
        'anchor_mask_percentage': anchor_mask_percentage,
        'anchor_mask_active': anchor_mask_active,
        'anchor_mask_total': anchor_mask_total,
    }
    return info


def _render_display_main(
    scene_model: BaseSceneModel,
    model: OctreeGS_3DGS,
    camera: Camera,
    resolution: Tuple[int, int],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    levels_rgb: List[int],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
) -> Dict[str, Any]:
    render_camera = camera
    rgb_image: Optional[torch.Tensor] = None
    if camera_name is not None:
        rgb_image = scene_model._get_snapshot(camera_name)

    if rgb_image is None:
        rgb_image = render_rgb_from_octree_gs(
            model=model,
            camera=render_camera,
            resolution=resolution,
            levels=levels_rgb,
            return_info=False,
        )
        if camera_name is not None:
            snapshot = rgb_image.detach().cpu()
            scene_model._put_snapshot(camera_name, snapshot)

    rgb_composed = BaseSceneModel._apply_camera_overlays(
        image=rgb_image,
        display_cameras=display_cameras,
        render_at_camera=render_camera,
        resolution=resolution,
    )
    title_value = title if title is not None else ""
    return {
        'dataset_name': dataset_name,
        'scene_name': scene_name,
        'method_name': method_name,
        'debugger_enabled': False,
        'rgb_image': rgb_composed,
        'title': title_value,
    }


def _render_display_debugger(
    scene_model: BaseSceneModel,
    model: OctreeGS_3DGS,
    camera: Camera,
    resolution: Tuple[int, int],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    levels_rgb: List[int],
    levels_density: List[int],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    total_levels: int,
) -> Dict[str, Any]:
    camera = camera.to(model.get_anchor.device)
    rgb_image: Optional[torch.Tensor] = None
    if camera_name is not None:
        rgb_image = scene_model._get_snapshot(camera_name)

    if rgb_image is None:
        rgb_image, info = render_rgb_from_octree_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            levels=levels_rgb,
            return_info=True,
        )
        if camera_name is not None:
            snapshot = rgb_image.detach().cpu()
            scene_model._put_snapshot(camera_name, snapshot)
    else:
        info = _compute_debugger_info(
            model=model,
            camera=camera,
            resolution=resolution,
            total_levels=total_levels,
        )
    gaussian_counts_per_level: Dict[int, int] = info['gaussian_counts_per_level']
    assert isinstance(
        gaussian_counts_per_level, dict
    ), f"gaussian_counts_per_level must be dict, got {type(gaussian_counts_per_level)=}"
    assert len(gaussian_counts_per_level) == total_levels, (
        f"gaussian_counts_per_level length mismatch: "
        f"expected {total_levels}, got {len(gaussian_counts_per_level)}"
    )
    for level in range(total_levels):
        assert (
            level in gaussian_counts_per_level
        ), f"gaussian_counts_per_level missing level {level}"
        assert isinstance(
            gaussian_counts_per_level[level], int
        ), f"gaussian count for level {level} must be int, got {type(gaussian_counts_per_level[level])=}"
    total_gaussians = info['total_gaussians']
    assert isinstance(
        total_gaussians, int
    ), f"total_gaussians must be int, got {type(total_gaussians)=}"
    assert 'anchor_mask_percentage' in info
    anchor_mask_percentage = info['anchor_mask_percentage']
    assert isinstance(
        anchor_mask_percentage, float
    ), f"anchor_mask_percentage must be float, got {type(anchor_mask_percentage)=}"

    assert 'anchor_mask_active' in info
    anchor_mask_active = info['anchor_mask_active']
    assert isinstance(
        anchor_mask_active, int
    ), f"anchor_mask_active must be int, got {type(anchor_mask_active)=}"

    assert 'anchor_mask_total' in info
    anchor_mask_total = info['anchor_mask_total']
    assert isinstance(
        anchor_mask_total, int
    ), f"anchor_mask_total must be int, got {type(anchor_mask_total)=}"

    rgb_composed = BaseSceneModel._apply_camera_overlays(
        image=rgb_image,
        display_cameras=display_cameras,
        render_at_camera=camera,
        resolution=resolution,
    )

    density_image = render_density_from_octree_gs(
        model=model,
        camera=camera,
        resolution=resolution,
        levels=levels_density,
        return_info=False,
        density_color=(0.0, 0.0, 1.0),
        uniform_scale=0.02,
    )
    density_composed = BaseSceneModel._apply_camera_overlays(
        image=density_image,
        display_cameras=display_cameras,
        render_at_camera=camera,
        resolution=resolution,
    )

    rgb_level_images: List[torch.Tensor] = []
    density_level_images: List[torch.Tensor] = []
    for level in range(total_levels):
        level_rgb_image = render_rgb_from_octree_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            levels=[level],
            return_info=False,
        )
        rgb_level_images.append(level_rgb_image)

        level_density_image = render_density_from_octree_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            levels=[level],
            return_info=False,
            density_color=(0.0, 0.0, 1.0),
            uniform_scale=0.02,
        )
        density_level_images.append(level_density_image)

    title_value = title if title is not None else ""
    return {
        'dataset_name': dataset_name,
        'scene_name': scene_name,
        'method_name': method_name,
        'debugger_enabled': True,
        'selected_levels_rgb': levels_rgb,
        'selected_levels_density': levels_density,
        'total_levels': total_levels,
        'gaussian_counts_per_level': gaussian_counts_per_level,
        'total_gaussians': total_gaussians,
        'anchor_mask_percentage': anchor_mask_percentage,
        'anchor_mask_active': anchor_mask_active,
        'anchor_mask_total': anchor_mask_total,
        'rgb_image': rgb_composed,
        'density_image': density_composed,
        'rgb_level_images': rgb_level_images,
        'density_level_images': density_level_images,
        'title': title_value,
    }
