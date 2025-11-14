"""Blender-based mesh rendering helpers parallel to the PyTorch3D stack."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch

import bpy
from mathutils import Matrix

from utils.input_checks.check_camera import (
    check_camera_extrinsics,
    check_camera_intrinsics,
)
from utils.three_d.camera.conventions import apply_coordinate_transform
from utils.three_d.camera.scaling import scale_intrinsics


def render_rgb_from_mesh_blender(
    mesh_collection_name: str,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str = 'opengl',
    background: Tuple[int, int, int] = (0, 0, 0),
    engine: str = 'CYCLES',
    device: str = 'GPU',
    view_layer_name: str = 'View Layer',
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Render RGB (and optional mask) using Blender's renderer."""
    assert isinstance(resolution, tuple) and len(resolution) == 2
    context = _build_render_context_blender(
        mesh_collection_name=mesh_collection_name,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        resolution=resolution,
        convention=convention,
        background=background,
        engine=engine,
        device=device,
        view_layer_name=view_layer_name,
        return_mask=return_mask,
    )
    try:
        return _execute_render_and_extract_blender(context)
    finally:
        _teardown_render_context_blender(context)


def _build_render_context_blender(
    mesh_collection_name: str,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str,
    background: Tuple[int, int, int],
    engine: str,
    device: str,
    view_layer_name: str,
    return_mask: bool,
) -> Dict[str, object]:
    scene, view_layer = _resolve_scene_and_view_layer_blender(view_layer_name)
    target_objects = _get_collection_objects_blender(mesh_collection_name)
    object_states = _prepare_objects_for_render_blender(target_objects, pass_index=1)
    camera_obj = _create_camera_from_parameters_blender(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        resolution=resolution,
        convention=convention,
    )
    previous_camera = scene.camera
    scene.camera = camera_obj

    render_state = _configure_render_settings_blender(scene, resolution)
    world_state = _configure_world_background_blender(scene, background)
    layer_state = _configure_layer_mask_blender(view_layer, return_mask)
    engine_state = _configure_render_engine_blender(scene, engine, device)

    return {
        'scene': scene,
        'view_layer': view_layer,
        'camera_obj': camera_obj,
        'previous_camera': previous_camera,
        'object_states': object_states,
        'render_state': render_state,
        'world_state': world_state,
        'layer_state': layer_state,
        'engine_state': engine_state,
        'resolution': resolution,
        'view_layer_name': view_layer_name,
        'return_mask': return_mask,
    }


def _resolve_scene_and_view_layer_blender(
    view_layer_name: str,
) -> Tuple['bpy.types.Scene', 'bpy.types.ViewLayer']:
    scene = bpy.context.scene
    view_layer = scene.view_layers.get(view_layer_name)
    if view_layer is None:
        raise ValueError(f"View layer '{view_layer_name}' not found in current scene")
    return scene, view_layer


def _get_collection_objects_blender(
    collection_name: str,
) -> Sequence['bpy.types.Object']:
    if collection_name not in bpy.data.collections:
        raise ValueError(
            f"Collection '{collection_name}' does not exist in this .blend file"
        )
    collection = bpy.data.collections[collection_name]
    objects = [obj for obj in collection.objects if obj.type == 'MESH']
    if not objects:
        raise RuntimeError(
            f"Collection '{collection_name}' does not contain mesh objects"
        )
    return objects


def _prepare_objects_for_render_blender(
    target_objects: Sequence['bpy.types.Object'],
    pass_index: int,
) -> Dict[str, Tuple[bool, float]]:
    states: Dict[str, Tuple[bool, float]] = {}
    target_names = {obj.name for obj in target_objects}
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        states[obj.name] = (obj.hide_render, obj.pass_index)
        if obj.name in target_names:
            obj.hide_render = False
            obj.pass_index = pass_index
        else:
            obj.hide_render = True
    return states


def _create_camera_from_parameters_blender(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    resolution: Tuple[int, int],
    convention: str,
) -> 'bpy.types.Object':
    intrinsics = check_camera_intrinsics(intrinsics).clone()
    extrinsics = check_camera_extrinsics(extrinsics).clone()

    scale_intrinsics(intrinsics=intrinsics, resolution=resolution, inplace=True)

    extrinsics_std = apply_coordinate_transform(
        extrinsics=extrinsics,
        source_convention=convention,
        target_convention='standard',
    )

    camera_data = bpy.data.cameras.new(name='mesh_camera_blender')
    camera_obj = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)

    image_height, image_width = resolution
    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    cx = float(intrinsics[0, 2].item())
    cy = float(intrinsics[1, 2].item())

    sensor_width = camera_data.sensor_width
    sensor_height = camera_data.sensor_height

    camera_data.lens = fx * sensor_width / float(image_width)

    sensor_aspect = (fy * sensor_height / image_height) / (
        fx * sensor_width / image_width
    )
    camera_data.sensor_fit = 'HORIZONTAL' if sensor_aspect <= 1.0 else 'VERTICAL'

    camera_data.shift_x = (cx - image_width / 2.0) / image_width
    camera_data.shift_y = (image_height / 2.0 - cy) / image_height

    camera_obj.matrix_world = _torch_to_matrix_blender(extrinsics_std)
    return camera_obj


def _torch_to_matrix_blender(tensor: torch.Tensor) -> 'Matrix':
    array = tensor.detach().cpu().numpy()
    if array.shape != (4, 4):
        raise ValueError('Expected a 4x4 transform matrix')
    return Matrix(array.tolist())


def _configure_render_settings_blender(
    scene: 'bpy.types.Scene',
    resolution: Tuple[int, int],
) -> Dict[str, Union[int, float, str]]:
    render_settings = scene.render
    previous = {
        'resolution_x': render_settings.resolution_x,
        'resolution_y': render_settings.resolution_y,
        'percentage': render_settings.resolution_percentage,
        'aspect_x': render_settings.pixel_aspect_x,
        'aspect_y': render_settings.pixel_aspect_y,
        'color_mode': render_settings.image_settings.color_mode,
    }
    render_settings.resolution_x = resolution[1]
    render_settings.resolution_y = resolution[0]
    render_settings.resolution_percentage = 100
    render_settings.pixel_aspect_x = 1.0
    render_settings.pixel_aspect_y = 1.0
    render_settings.image_settings.color_mode = 'RGBA'
    return previous


def _configure_world_background_blender(
    scene: 'bpy.types.Scene',
    background: Tuple[int, int, int],
) -> Dict[str, object]:
    prev_world = scene.world
    created_world = False
    if prev_world is None:
        scene.world = bpy.data.worlds.new('mesh_world_blender')
        created_world = True
    world = scene.world
    previous_state: Dict[str, object] = {
        'previous_world': prev_world,
        'world': world,
        'created_world': created_world,
        'use_nodes': world.use_nodes,
        'color': tuple(world.color),
    }
    world.use_nodes = False
    world.color = tuple(channel / 255.0 for channel in background)
    return previous_state


def _configure_layer_mask_blender(
    view_layer: 'bpy.types.ViewLayer',
    enable_mask: bool,
) -> bool:
    previous = view_layer.use_pass_object_index
    view_layer.use_pass_object_index = enable_mask
    return previous


def _configure_render_engine_blender(
    scene: 'bpy.types.Scene',
    engine: str,
    device: str,
) -> Tuple[str, str, str]:
    engine_normalized = engine.upper()
    if engine_normalized == 'EEVEE':
        engine_normalized = 'BLENDER_EEVEE'
    previous_engine = scene.render.engine
    scene.render.engine = engine_normalized

    previous_device = ''
    previous_compute = ''
    if engine_normalized == 'CYCLES':
        previous_device = scene.cycles.device
        scene.cycles.device = 'GPU' if device.upper() == 'GPU' else 'CPU'
        cycles_addon = bpy.context.preferences.addons.get('cycles')
        if cycles_addon:
            prefs = cycles_addon.preferences
            previous_compute = prefs.compute_device_type
            if device.upper() == 'GPU':
                for candidate in ('OPTIX', 'CUDA', 'HIP', 'METAL'):
                    if candidate in prefs.get_devices():
                        prefs.compute_device_type = candidate
                        break
            else:
                prefs.compute_device_type = 'NONE'
    return previous_engine, previous_device, previous_compute


def _execute_render_and_extract_blender(
    context: Dict[str, object],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    bpy.ops.render.render(write_still=False, use_viewport=False)
    image = bpy.data.images['Render Result']
    resolution = context['resolution']  # type: ignore[index]
    rgb = _extract_combined_result_blender(image, resolution)  # type: ignore[arg-type]
    if context['return_mask']:
        mask = _extract_object_index_pass_blender(
            image=image,
            view_layer_name=context['view_layer_name'],  # type: ignore[index]
            resolution=resolution,  # type: ignore[arg-type]
            threshold=0.5,
        )
        return rgb, mask
    return rgb


def _extract_combined_result_blender(
    image: 'bpy.types.Image',
    resolution: Tuple[int, int],
) -> torch.Tensor:
    image_height, image_width = resolution
    expected_len = image_width * image_height * 4
    pixels = np.array(image.pixels[:], dtype=np.float32)
    if pixels.size != expected_len:
        raise RuntimeError("Combined render result has unexpected size")
    pixels = pixels.reshape((image_height, image_width, 4))
    pixels = np.flip(pixels, axis=0)
    rgb = pixels[:, :, :3]
    rgb = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return rgb.clamp(0.0, 1.0)


def _extract_object_index_pass_blender(
    image: 'bpy.types.Image',
    view_layer_name: str,
    resolution: Tuple[int, int],
    threshold: float = 0.5,
) -> torch.Tensor:
    image_height, image_width = resolution
    slot = image.render_slots[image.render_slots.active_index]
    layer = slot.layers.get(view_layer_name)
    if layer is None:
        raise RuntimeError(
            f"View layer '{view_layer_name}' not present in render result"
        )
    index_pass = layer.passes.get('IndexOB')
    if index_pass is None:
        raise RuntimeError(
            "Object Index pass was not rendered. Enable `use_pass_object_index`."
        )
    array = np.array(index_pass.rect, dtype=np.float32)
    channels = array.size // (image_height * image_width)
    array = array.reshape((image_height, image_width, channels))
    array = np.flip(array, axis=0)
    mask = (array[:, :, 0] > threshold).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(mask))


def _teardown_render_context_blender(context: Dict[str, object]) -> None:
    scene = context['scene']  # type: ignore[index]
    view_layer = context['view_layer']  # type: ignore[index]
    _restore_objects_after_render_blender(context['object_states'])  # type: ignore[arg-type]
    scene.camera = context['previous_camera']  # type: ignore[index]
    bpy.data.objects.remove(context['camera_obj'], do_unlink=True)  # type: ignore[arg-type]
    engine_state = context['engine_state']  # type: ignore[index]
    _restore_render_engine_blender(
        scene=scene,
        previous_engine=engine_state[0],  # type: ignore[index]
        previous_device=engine_state[1],  # type: ignore[index]
        previous_compute=engine_state[2],  # type: ignore[index]
    )
    _restore_render_settings_blender(scene, context['render_state'])  # type: ignore[arg-type]
    _restore_world_background_blender(scene, context['world_state'])  # type: ignore[arg-type]
    _restore_layer_mask_blender(view_layer, context['layer_state'])  # type: ignore[arg-type]


def _restore_objects_after_render_blender(
    states: Dict[str, Tuple[bool, float]],
) -> None:
    for name, (hide_render, pass_index) in states.items():
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        obj.hide_render = hide_render
        obj.pass_index = pass_index


def _restore_render_engine_blender(
    scene: 'bpy.types.Scene',
    previous_engine: str,
    previous_device: str,
    previous_compute: str,
) -> None:
    scene.render.engine = previous_engine
    if previous_engine == 'CYCLES' and previous_device:
        scene.cycles.device = previous_device
        cycles_addon = bpy.context.preferences.addons.get('cycles')
        if cycles_addon and previous_compute:
            cycles_addon.preferences.compute_device_type = previous_compute


def _restore_render_settings_blender(
    scene: 'bpy.types.Scene',
    previous: Dict[str, Union[int, float, str]],
) -> None:
    render_settings = scene.render
    render_settings.resolution_x = int(previous['resolution_x'])
    render_settings.resolution_y = int(previous['resolution_y'])
    render_settings.resolution_percentage = int(previous['percentage'])
    render_settings.pixel_aspect_x = float(previous['aspect_x'])
    render_settings.pixel_aspect_y = float(previous['aspect_y'])
    render_settings.image_settings.color_mode = str(previous['color_mode'])


def _restore_world_background_blender(
    scene: 'bpy.types.Scene',
    state: Dict[str, object],
) -> None:
    world = state['world']
    if isinstance(world, bpy.types.World):
        color = state['color']
        if isinstance(color, tuple):
            world.color = color
        use_nodes = state['use_nodes']
        if isinstance(use_nodes, bool):
            world.use_nodes = use_nodes
    created_world = bool(state['created_world'])
    previous_world = state['previous_world']
    if created_world and isinstance(world, bpy.types.World):
        bpy.data.worlds.remove(world, do_unlink=True)
    if isinstance(previous_world, bpy.types.World) or previous_world is None:
        scene.world = previous_world


def _restore_layer_mask_blender(
    view_layer: 'bpy.types.ViewLayer',
    previous: bool,
) -> None:
    view_layer.use_pass_object_index = previous
