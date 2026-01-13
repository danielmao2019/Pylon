from typing import Any, Dict, List, Optional, Tuple

import torch

from data.structures.three_d.camera.camera import Camera
from models.three_d.base import BaseSceneModel
from models.three_d.lapis_gs.render import (
    render_density_from_lapis_gs,
    render_rgb_from_lapis_gs,
)
from models.three_d.original_3dgs.loader import GaussianModel as GaussianModel3D


def render_display(
    scene_model: BaseSceneModel,
    camera: Camera,
    resolution: Tuple[int, int],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    debugger_enabled: bool,
    selected_layers_rgb: Optional[List[Any]],
    selected_layers_density: Optional[List[Any]],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    device: Optional[torch.device],
) -> Dict[str, Any]:
    # Input validation
    assert isinstance(scene_model, BaseSceneModel), f"{type(scene_model)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert isinstance(dataset_name, str) and dataset_name, f"{dataset_name=}"
    assert isinstance(scene_name, str) and scene_name, f"{scene_name=}"
    assert isinstance(method_name, str) and method_name, f"{method_name=}"
    assert isinstance(debugger_enabled, bool), f"{type(debugger_enabled)=}"
    assert resolution and isinstance(resolution, tuple) and len(resolution) == 2
    assert camera_name is None or isinstance(camera_name, str), f"{type(camera_name)=}"
    assert device is None or isinstance(device, torch.device), f"{type(device)=}"
    if display_cameras is not None:
        assert isinstance(display_cameras, list), f"{type(display_cameras)=}"
        assert all(isinstance(cam, Camera) for cam in display_cameras)

    resolved_device = device if device is not None else scene_model.device
    camera = camera.to(resolved_device)

    model = scene_model.model
    assert isinstance(
        model, GaussianModel3D
    ), f"Expected GaussianModel for lapis_gs, got {type(model)}"
    assert hasattr(model, 'split_points')
    assert hasattr(model, 'layer_names')

    split_points = model.split_points
    layer_names = model.layer_names
    num_layers = len(split_points) - 1
    assert num_layers > 0, "LapisGS model must have at least one layer"

    full_layers = list(range(num_layers))
    cache_allowed = (
        camera_name is not None
        and not debugger_enabled
        and selected_layers_rgb == full_layers
    )
    if not debugger_enabled:
        selected_layers_rgb = full_layers
        selected_layers_density = full_layers
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
            layers_rgb=selected_layers_rgb,
            layers_density=selected_layers_density,
            camera_name=camera_name,
            display_cameras=display_cameras,
            title=title,
            device=resolved_device,
            layer_names=layer_names,
        )

    return _render_display_main(
        scene_model=scene_model,
        model=model,
        camera=camera,
        resolution=resolution,
        dataset_name=dataset_name,
        scene_name=scene_name,
        method_name=method_name,
        layers_rgb=selected_layers_rgb,
        camera_name=camera_name,
        display_cameras=display_cameras,
        title=title,
        device=resolved_device,
    )


def _render_display_main(
    scene_model: BaseSceneModel,
    model: GaussianModel3D,
    camera: Camera,
    resolution: Tuple[int, int],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    layers_rgb: List[int],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    device: torch.device,
) -> Dict[str, Any]:
    camera = camera.to(device)
    rgb_image: Optional[torch.Tensor] = None
    if camera_name is not None:
        rgb_image = scene_model._get_snapshot(camera_name)

    if rgb_image is None:
        rgb_image = render_rgb_from_lapis_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            layers=layers_rgb,
            return_info=False,
            device=device,
        )
        if camera_name is not None:
            snapshot = rgb_image.detach().cpu()
            scene_model._put_snapshot(camera_name, snapshot)

    rgb_composed = BaseSceneModel._apply_camera_overlays(
        image=rgb_image,
        display_cameras=display_cameras,
        render_at_camera=camera,
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
    model: GaussianModel3D,
    camera: Camera,
    resolution: Tuple[int, int],
    dataset_name: str,
    scene_name: str,
    method_name: str,
    layers_rgb: List[int],
    layers_density: List[int],
    camera_name: Optional[str],
    display_cameras: Optional[List[Camera]],
    title: Optional[str],
    device: torch.device,
    layer_names: List[str],
) -> Dict[str, Any]:
    camera = camera.to(device)
    num_layers = len(layer_names)
    rgb_image: Optional[torch.Tensor] = None
    if camera_name is not None:
        rgb_image = scene_model._get_snapshot(camera_name)

    if rgb_image is None:
        rgb_image, info = render_rgb_from_lapis_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            layers=layers_rgb,
            return_info=True,
            device=device,
        )
        gaussian_counts_per_layer = info['gaussian_counts_per_layer']
        total_gaussians = info['total_gaussians']
        if camera_name is not None:
            snapshot = rgb_image.detach().cpu()
            scene_model._put_snapshot(camera_name, snapshot)
    else:
        rgb_image, gaussian_counts_per_layer, total_gaussians = (
            rgb_image,
            None,
            None,
        )

    rgb_composed = BaseSceneModel._apply_camera_overlays(
        image=rgb_image,
        display_cameras=display_cameras,
        render_at_camera=camera,
        resolution=resolution,
    )

    density_image = render_density_from_lapis_gs(
        model=model,
        camera=camera,
        resolution=resolution,
        layers=layers_density,
        return_info=False,
        density_color=(0.0, 0.0, 1.0),
        uniform_scale=0.02,
        device=device,
    )
    density_composed = BaseSceneModel._apply_camera_overlays(
        image=density_image,
        display_cameras=display_cameras,
        render_at_camera=camera,
        resolution=resolution,
    )

    rgb_layer_images: List[torch.Tensor] = []
    density_layer_images: List[torch.Tensor] = []
    for layer in range(num_layers):
        layer_rgb_image = render_rgb_from_lapis_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            layers=[layer],
            return_info=False,
            device=device,
        )
        rgb_layer_images.append(layer_rgb_image)

        layer_density_image = render_density_from_lapis_gs(
            model=model,
            camera=camera,
            resolution=resolution,
            layers=[layer],
            return_info=False,
            density_color=(0.0, 0.0, 1.0),
            uniform_scale=0.02,
            device=device,
        )
        density_layer_images.append(layer_density_image)

    title_value = title if title is not None else ""
    return {
        'dataset_name': dataset_name,
        'scene_name': scene_name,
        'method_name': method_name,
        'debugger_enabled': True,
        'selected_layers_rgb': layers_rgb,
        'selected_layers_density': layers_density,
        'num_layers': num_layers,
        'layer_names': layer_names,
        'gaussian_counts_per_layer': gaussian_counts_per_layer,
        'total_gaussians': total_gaussians,
        'rgb_image': rgb_composed,
        'density_image': density_composed,
        'rgb_layer_images': rgb_layer_images,
        'density_layer_images': density_layer_images,
        'title': title_value,
    }
