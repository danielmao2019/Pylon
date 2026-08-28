# `models/three_d/meshes/render/` code skeleton

## Code implementation structure

`models/three_d/meshes/render/__init__.py`

```text
__init__.py
├── from models.three_d.meshes.render.core import render_rgb_from_mesh, render_soft_mask_from_mesh
├── from models.three_d.meshes.render.display import render_display
├── from models.three_d.meshes.render.shading import compute_sh_shading
└── from models.three_d.meshes.render.uv_texture import render_uv_texture_aligned
```

`models/three_d/meshes/render/shading.py`

```text
shading.py
├── import math
├── import torch
└── def compute_sh_shading(normals: torch.Tensor, sh_coefficients: torch.Tensor) -> torch.Tensor
    ├── # Evaluates spherical-harmonic shading over surface normals, at whatever band count sh_coefficients carries.
    ├── def _validate_inputs [local]
    │   ├── impls assert normals is a float32 torch tensor shaped [..., N, 3]
    │   └── impls assert sh_coefficients is a same-device float32 tensor shaped [..., B * 3] whose B is a positive perfect square
    ├── calls _validate_inputs
    ├── impls order = the integer square root of sh_coefficients' band count
    ├── impls evaluate the spherical-harmonic basis over the normals up to that order  # impls-node-one-step:skip
    ├── impls contract the basis against sh_coefficients over the bands
    └── return  # the per-normal RGB shading the caller multiplies its albedo by
```

`models/three_d/meshes/render/core.py`

```text
core.py
├── from math import log
├── from typing import Optional, Tuple, Union
├── import numpy as np
├── import torch
├── from pytorch3d.renderer import BlendParams, MeshRasterizer, MeshRenderer, OrthographicCameras, PerspectiveCameras, PointLights, RasterizationSettings, SoftPhongShader, SoftSilhouetteShader
├── from pytorch3d.renderer.cameras import CamerasBase
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.mesh.convert import mesh_to_pytorch3d
├── from data.structures.three_d.mesh.mesh import Mesh
├── @torch.no_grad()
├── def render_rgb_from_mesh(mesh: Mesh, camera: Camera, resolution: Optional[Tuple[int, int]] = None, background: Tuple[int, int, int] = (0, 0, 0), return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders an RGB image (and optionally a validity mask) from a triangle Mesh using PyTorch3D.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert mesh is a Mesh
│   │   ├── impls assert camera is a Camera
│   │   ├── impls assert resolution is None or a length-two tuple of positive int/scalar-tensor values
│   │   ├── impls assert background is a length-three tuple of integer RGB channels in [0, 255]
│   │   └── impls assert return_mask is bool
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── if resolution is None
│   │   │   └── impls resolution = camera.intrinsics.resolution
│   │   ├── for each axis value
│   │   │   └── impls convert int/scalar-tensor values to ints
│   │   └── return resolution
│   ├── calls _normalize_inputs
│   ├── assert CUDA is available
│   ├── impls device = torch.device("cuda")
│   ├── calls mesh_to_pytorch3d(mesh=mesh, device=device, dtype=torch.float32)
│   ├── calls _prepare_cameras(camera=camera, resolution=resolution, device=device)
│   ├── calls _build_rasterizer(cameras=cameras, resolution=resolution, blur_radius=0.0, faces_per_pixel=1, clip_barycentric_coords=False)  # one face per pixel with zero blur, so this render returns a hard image
│   ├── calls _build_phong_shader(cameras=cameras, device=device, background_color=background)
│   ├── calls MeshRenderer(rasterizer=rasterizer, shader=shader)  # -> renderer
│   ├── impls images = renderer(meshes)                           # the shader's own [1, H, W, 4] blend
│   ├── impls rgb = images[0, :, :, :3] permuted to [3, H, W]
│   ├── if return_mask
│   │   ├── impls fragments = rasterizer(meshes)                       # coverage comes from the rasterization face-index sentinel, so background-colored faces stay covered
│   │   ├── impls valid_mask = fragments.pix_to_face[0, :, :, 0] >= 0  # a background pixel carries the -1 sentinel
│   │   ├── impls render_result = (rgb, valid_mask)
│   │   └── return render_result
│   └── return rgb  # [3, H, W] float32, the shaded image alone
├── def render_soft_mask_from_mesh(mesh: Mesh, camera: Camera, blend_sigma: float, blend_gamma: float, faces_per_pixel: int, coverage_threshold: float, resolution: Optional[Tuple[int, int]] = None) -> torch.Tensor
│   ├── # Renders a continuous coverage mask from a triangle Mesh using PyTorch3D, differentiable in the mesh and in the camera it is rendered through.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert mesh is a Mesh
│   │   ├── impls assert camera is a Camera
│   │   ├── impls assert blend_sigma is a positive float
│   │   ├── impls assert blend_gamma is a positive float
│   │   ├── impls assert faces_per_pixel is a positive int
│   │   ├── impls assert coverage_threshold is a float in (0, 1)
│   │   └── impls assert resolution is None or a length-two tuple of positive int/scalar-tensor values
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── if resolution is None
│   │   │   └── impls resolution = camera.intrinsics.resolution
│   │   ├── for each axis value
│   │   │   └── impls convert int/scalar-tensor values to ints
│   │   └── return resolution
│   ├── calls _normalize_inputs
│   ├── assert CUDA is available
│   ├── impls device = torch.device("cuda")
│   ├── calls mesh_to_pytorch3d(mesh=mesh, device=device, dtype=torch.float32)       # the mesh's world-space vertices as they are, placing them before a camera being the camera's work
│   ├── calls _prepare_cameras(camera=camera, resolution=resolution, device=device)  # whatever coordinate system the camera names, it reaches PyTorch3D in the one PyTorch3D names
│   ├── impls blur_radius = log(1 / coverage_threshold - 1) * blend_sigma            # how far past its own edge a face is rasterized, derived here from the two values that fix it so the blur ends exactly at coverage_threshold
│   ├── calls _build_rasterizer(cameras=cameras, resolution=resolution, blur_radius=blur_radius, faces_per_pixel=faces_per_pixel, clip_barycentric_coords=True)
│   ├── calls _build_silhouette_shader(blend_sigma=blend_sigma, blend_gamma=blend_gamma)
│   ├── calls MeshRenderer(rasterizer=rasterizer, shader=shader)  # -> renderer
│   ├── impls images = renderer(meshes)                           # the shader's own [1, H, W, 4] blend
│   ├── impls mask = images[0, :, :, 3]                           # [H, W] float32 in [0, 1], the alpha channel
│   └── return mask
├── def _prepare_cameras(camera: Camera, resolution: Tuple[int, int], device: torch.device) -> CamerasBase
│   ├── # Builds the PyTorch3D camera a rasterizer carries world-space vertices through, so both coordinate system changes a render is — world to camera, then camera to normalized device coordinates — are this camera's own.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert camera is a Camera
│   │   ├── impls assert resolution is a length-two tuple of positive ints
│   │   └── impls assert device is a torch.device
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── impls camera = camera.to(device=device, intr_convention="pytorch3d", extr_convention="pytorch3d").scale_intrinsics(resolution=resolution)
│   │   └── return camera
│   ├── calls _normalize_inputs
│   ├── impls R, T = the rotation and translation blocks of that camera's extrinsics.w2c  # the world-to-camera change stays on the camera  # impls-node-one-step:skip
│   ├── impls R = that rotation block transposed                                          # PyTorch3D composes a point with its camera as a row vector, so the world-to-camera rotation it reads is the transpose of the column-vector one w2c holds
│   ├── impls zfar = np.float32 maximum  # keeps distant geometry from the shader's finite far-plane overflow
│   ├── if camera.intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   ├── calls PerspectiveCameras(R=R, T=T, focal_length=((camera.intrinsics.fx, camera.intrinsics.fy),), principal_point=((camera.intrinsics.cx, camera.intrinsics.cy),), in_ndc=True, device=device)  # pinhole intrinsics are already in PyTorch3D's normalized device frame
│   │   └── return  # that perspective camera
│   ├── if camera.intrinsics.model == "ortho"
│   │   ├── calls OrthographicCameras(R=R, T=T, focal_length=((camera.intrinsics.fx, camera.intrinsics.fy),), principal_point=((camera.intrinsics.cx, camera.intrinsics.cy),), in_ndc=True, device=device)  # weak-perspective intrinsics are already in PyTorch3D's normalized device frame
│   │   └── return  # that orthographic camera
│   └── assert 0, "Should not reach here."
├── def _build_rasterizer(cameras: CamerasBase, resolution: Tuple[int, int], blur_radius: float, faces_per_pixel: int, clip_barycentric_coords: bool) -> MeshRasterizer
│   ├── # Builds a MeshRasterizer for the given cameras and resolution, how far it blurs, how many faces a pixel keeps and whether it clips barycentric coordinates each being the render's own.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert cameras is a CamerasBase
│   │   ├── impls assert resolution is a length-two tuple of positive ints
│   │   ├── impls assert blur_radius is a float
│   │   ├── impls assert faces_per_pixel is a positive int
│   │   └── impls assert clip_barycentric_coords is bool
│   ├── calls _validate_inputs
│   ├── calls RasterizationSettings(image_size=resolution, blur_radius=blur_radius, faces_per_pixel=faces_per_pixel, cull_backfaces=False, clip_barycentric_coords=clip_barycentric_coords, bin_size=0)  # -> raster_settings; the culling is the module's own, and bin_size 0 holds every render to the naive kernel, the one that reproduces itself between runs
│   ├── calls MeshRasterizer(cameras=cameras, raster_settings=raster_settings)  # -> rasterizer
│   └── return rasterizer
├── def _build_phong_shader(cameras: CamerasBase, device: torch.device, background_color: Tuple[int, int, int]) -> SoftPhongShader
│   ├── # Builds a flat-ambient SoftPhongShader with the given normalized background color.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert cameras is a CamerasBase
│   │   ├── impls assert device is a torch.device
│   │   └── impls assert background_color is a length-three tuple of integer RGB channels in [0, 255]
│   ├── calls _validate_inputs
│   ├── impls background = each background_color channel over 255
│   ├── calls PointLights(device=device, location=the origin, ambient_color=one, diffuse_color=zero, specular_color=zero)  # -> lights; a flat ambient light, so the shader returns the albedo it was handed
│   ├── calls BlendParams(background_color=background)  # -> blend_params
│   ├── calls SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)  # -> shader
│   └── return shader
└── def _build_silhouette_shader(blend_sigma: float, blend_gamma: float) -> SoftSilhouetteShader
    ├── # Builds PyTorch3D's own silhouette shader from the library's soft-rasterization blend.
    ├── def _validate_inputs [local]
    │   ├── impls assert blend_sigma is a positive float
    │   └── impls assert blend_gamma is a positive float
    ├── calls _validate_inputs
    ├── calls BlendParams(sigma=blend_sigma, gamma=blend_gamma)  # -> blend_params
    ├── calls SoftSilhouetteShader(blend_params=blend_params)    # -> shader
    └── return shader
```

`models/three_d/meshes/render/core_blender.py`

```text
core_blender.py
├── from typing import Dict, Optional, Sequence, Tuple, Union
├── import bpy
├── import numpy as np
├── import torch
├── from mathutils import Matrix
├── from data.structures.three_d.camera.camera import Camera
├── def render_rgb_from_mesh_blender(mesh_collection_name: str, camera: Camera, resolution: Optional[Tuple[int, int]] = None, background: Tuple[int, int, int] = (0, 0, 0), engine: str = 'CYCLES', device: str = 'GPU', view_layer_name: str = 'View Layer', return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders RGB (and optional mask) of a named mesh collection using Blender's renderer, restoring scene state afterward.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert mesh_collection_name is a string
│   │   ├── impls assert camera is a Camera
│   │   ├── impls assert resolution is None or a length-two tuple of positive int/scalar-tensor values
│   │   ├── impls assert background is a length-three tuple of integer RGB channels in [0, 255]
│   │   ├── impls assert engine is a string
│   │   ├── impls assert device is a string
│   │   ├── impls assert view_layer_name is a string
│   │   └── impls assert return_mask is bool
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── if resolution is None
│   │   │   └── impls resolution = camera.intrinsics.resolution  # the camera's own h and w are two of its params, so what raster to render is already answered
│   │   ├── for each axis value
│   │   │   └── impls convert int/scalar-tensor values to ints
│   │   └── return resolution
│   ├── calls _normalize_inputs
│   ├── calls _build_render_context_blender(mesh_collection_name=mesh_collection_name, camera=camera, resolution=resolution, background=background, engine=engine, device=device, view_layer_name=view_layer_name, return_mask=return_mask)
│   ├── try
│   │   └── calls _execute_render_and_extract_blender(context)
│   └── finally
│       └── calls _teardown_render_context_blender(context)
├── def _build_render_context_blender(mesh_collection_name: str, camera: Camera, resolution: Tuple[int, int], background: Tuple[int, int, int], engine: str, device: str, view_layer_name: str, return_mask: bool) -> Dict[str, object]
│   ├── # Resolves scene/layer, prepares objects, sets the camera, and applies render/world/layer/engine settings, returning a saved-state context dict.
│   ├── calls _resolve_scene_and_view_layer_blender(view_layer_name)
│   ├── calls _get_collection_objects_blender(mesh_collection_name)
│   ├── calls _prepare_objects_for_render_blender(target_objects, pass_index=1)
│   ├── calls _create_camera_from_parameters_blender(camera=camera, resolution=resolution)
│   ├── calls _configure_render_settings_blender(scene, resolution)
│   ├── calls _configure_world_background_blender(scene, background)
│   ├── calls _configure_layer_mask_blender(view_layer, return_mask)
│   └── calls _configure_render_engine_blender(scene, engine, device)
├── def _resolve_scene_and_view_layer_blender(view_layer_name: str) -> Tuple['bpy.types.Scene', 'bpy.types.ViewLayer']
│   ├── # Returns the active scene and its named view layer, raising if the view layer is absent.
│   └── if view_layer is None
│       └── raise ValueError
├── def _get_collection_objects_blender(collection_name: str) -> Sequence['bpy.types.Object']
│   ├── # Returns the mesh objects of a named collection, raising if the collection or its mesh objects are missing.
│   ├── if collection_name not in bpy.data.collections
│   │   └── raise ValueError
│   └── if not objects
│       └── raise RuntimeError
├── def _prepare_objects_for_render_blender(target_objects: Sequence['bpy.types.Object'], pass_index: int) -> Dict[str, Tuple[bool, float]]
│   ├── # Saves and overrides every mesh object's hide_render/pass_index so only the targets render with the given pass index.
│   └── for each obj in bpy.data.objects
│       ├── if obj.type != 'MESH'
│       │   └── continue
│       ├── if obj.name in target_names
│       │   ├── impls obj.hide_render = False
│       │   └── impls obj.pass_index = pass_index
│       └── else
│           └── impls obj.hide_render = True
├── def _create_camera_from_parameters_blender(camera: Camera, resolution: Tuple[int, int]) -> 'bpy.types.Object'
│   ├── # Creates a temporary Blender camera object whose scaled intrinsics and pose match the repo Camera at the render resolution.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert camera is a Camera
│   │   └── impls assert resolution is a length-two tuple of positive ints
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── impls camera = camera.to(intr_convention='standard', extr_convention='standard').scale_intrinsics(resolution=resolution)
│   │   └── return camera
│   ├── calls _normalize_inputs
│   ├── impls lens values come from camera.intrinsics
│   ├── impls shift values come from camera.intrinsics
│   └── calls _torch_to_matrix_blender(extrinsics)
├── def _torch_to_matrix_blender(tensor: torch.Tensor) -> 'Matrix'
│   ├── # Converts a 4x4 torch transform tensor into a mathutils Matrix, raising if the shape is wrong.
│   └── if array.shape != (4, 4)
│       └── raise ValueError
├── def _configure_render_settings_blender(scene: 'bpy.types.Scene', resolution: Tuple[int, int]) -> Dict[str, Union[int, float, str]]
│   ├── # Saves and overrides the scene render resolution/aspect/color-mode settings, returning the previous values.
│   ├── impls previous = the scene render's resolution_x and resolution_y, its resolution_percentage under "percentage", its pixel_aspect_x and pixel_aspect_y under "aspect_x" and "aspect_y", and its image color_mode  # impls-node-one-step:skip
│   ├── impls render_settings.resolution_x = resolution[1]  # resolution arrives as (height, width)
│   ├── impls render_settings.resolution_y = resolution[0]
│   ├── impls render_settings.resolution_percentage = 100
│   ├── impls render_settings.pixel_aspect_x = 1.0
│   ├── impls render_settings.pixel_aspect_y = 1.0
│   ├── impls render_settings.image_settings.color_mode = "RGBA"
│   └── return previous  # exactly the keys _restore_render_settings_blender puts back
├── def _configure_world_background_blender(scene: 'bpy.types.Scene', background: Tuple[int, int, int]) -> Dict[str, object]
│   ├── # Saves and overrides the scene world's flat background color, creating a world if none exists.
│   └── if prev_world is None
│       └── impls scene.world = bpy.data.worlds.new('mesh_world_blender'); created_world = True
├── def _configure_layer_mask_blender(view_layer: 'bpy.types.ViewLayer', enable_mask: bool) -> bool
│   ├── # Saves and sets the view layer's object-index pass flag, returning the previous value.
│   ├── impls previous = view_layer.use_pass_object_index
│   ├── impls view_layer.use_pass_object_index = enable_mask  # the pass _extract_object_index_pass_blender reads the mask out of
│   └── return previous
├── def _configure_render_engine_blender(scene: 'bpy.types.Scene', engine: str, device: str) -> Tuple[str, str, str]
│   ├── # Saves and sets the render engine plus Cycles device/compute-type, returning the previous engine/device/compute.
│   ├── def _validate_inputs [local]
│   │   ├── impls assert scene is a Blender Scene
│   │   ├── impls assert engine is a string
│   │   └── impls assert device is a string
│   ├── calls _validate_inputs
│   ├── def _normalize_inputs [local]
│   │   ├── impls engine = engine.upper(), remapping EEVEE to BLENDER_EEVEE
│   │   ├── impls device = device.upper()
│   │   └── return engine, device
│   ├── calls _normalize_inputs
│   └── if engine == 'CYCLES'
│       └── if cycles_addon
│           ├── if device == 'GPU'
│           │   └── for each candidate in ('OPTIX', 'CUDA', 'HIP', 'METAL')
│           │       └── if candidate in prefs.get_devices()
│           │           ├── impls prefs.compute_device_type = candidate
│           │           └── break
│           └── else
│               └── impls prefs.compute_device_type = 'NONE'
├── def _execute_render_and_extract_blender(context: Dict[str, object]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Triggers the Blender render and extracts the combined RGB and, if requested, the object-index mask.
│   ├── calls _extract_combined_result_blender(image, resolution)
│   └── if context['return_mask']
│       └── calls _extract_object_index_pass_blender(image=image, view_layer_name=context['view_layer_name'], resolution=resolution, threshold=0.5)
├── def _extract_combined_result_blender(image: 'bpy.types.Image', resolution: Tuple[int, int]) -> torch.Tensor
│   ├── # Reads the combined render-result pixels into a clamped [3, H, W] RGB torch tensor, raising on unexpected size.
│   └── if pixels.size != expected_len
│       └── raise RuntimeError
├── def _extract_object_index_pass_blender(image: 'bpy.types.Image', view_layer_name: str, resolution: Tuple[int, int], threshold: float = 0.5) -> torch.Tensor
│   ├── # Reads the object-index pass into a thresholded [H, W] float mask, raising if the layer or pass is missing.
│   ├── if layer is None
│   │   └── raise RuntimeError
│   └── if index_pass is None
│       └── raise RuntimeError
├── def _teardown_render_context_blender(context: Dict[str, object]) -> None
│   ├── # Restores all saved scene state and removes the temporary camera created for the render.
│   ├── calls _restore_objects_after_render_blender(context['object_states'])
│   ├── calls _restore_render_engine_blender(scene=scene, previous_engine=engine_state[0], previous_device=engine_state[1], previous_compute=engine_state[2])
│   ├── calls _restore_render_settings_blender(scene, context['render_state'])
│   ├── calls _restore_world_background_blender(scene, context['world_state'])
│   └── calls _restore_layer_mask_blender(view_layer, context['layer_state'])
├── def _restore_objects_after_render_blender(states: Dict[str, Tuple[bool, float]]) -> None
│   ├── # Restores each mesh object's saved hide_render/pass_index by name, skipping objects that no longer exist.
│   └── for each (name, (hide_render, pass_index)) in states
│       └── if obj is None
│           └── continue
├── def _restore_render_engine_blender(scene: 'bpy.types.Scene', previous_engine: str, previous_device: str, previous_compute: str) -> None
│   ├── # Restores the previous render engine and Cycles device/compute-type settings.
│   └── if previous_engine == 'CYCLES' and previous_device
│       └── if cycles_addon and previous_compute
│           └── impls cycles_addon.preferences.compute_device_type = previous_compute
├── def _restore_render_settings_blender(scene: 'bpy.types.Scene', previous: Dict[str, Union[int, float, str]]) -> None
│   ├── # Restores the previously saved scene render resolution/aspect/color-mode settings.
│   ├── impls render_settings.resolution_x = previous["resolution_x"], as an int
│   ├── impls render_settings.resolution_y = previous["resolution_y"], as an int
│   ├── impls render_settings.resolution_percentage = previous["percentage"], as an int
│   ├── impls render_settings.pixel_aspect_x = previous["aspect_x"], as a float
│   ├── impls render_settings.pixel_aspect_y = previous["aspect_y"], as a float
│   └── impls render_settings.image_settings.color_mode = previous["color_mode"], as a str
├── def _restore_world_background_blender(scene: 'bpy.types.Scene', state: Dict[str, object]) -> None
│   ├── # Restores the world's saved color/use_nodes, removes a temporary world if one was created, and restores the previous world.
│   ├── if isinstance(world, bpy.types.World)
│   │   ├── if isinstance(color, tuple)
│   │   │   └── impls world.color = color
│   │   └── if isinstance(use_nodes, bool)
│   │       └── impls world.use_nodes = use_nodes
│   ├── if created_world and isinstance(world, bpy.types.World)
│   │   └── impls bpy.data.worlds.remove(world, do_unlink=True)
│   └── if isinstance(previous_world, bpy.types.World) or previous_world is None
│       └── impls scene.world = previous_world
└── def _restore_layer_mask_blender(view_layer: 'bpy.types.ViewLayer', previous: bool) -> None
    ├── # Restores the view layer's previous object-index pass flag.
    └── impls view_layer.use_pass_object_index = previous
```

`models/three_d/meshes/render/display.py`

```text
display.py
├── from typing import Any, Dict, List, Optional, Tuple
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from models.three_d.base import BaseSceneModel
├── from models.three_d.meshes.render.core import render_rgb_from_mesh as render_rgb_from_mesh_func
└── def render_display(scene_model: BaseSceneModel, camera: Camera, resolution: Optional[Tuple[int, int]], camera_name: Optional[str], display_cameras: Optional[List[Camera]], title: Optional[str], device: Optional[torch.device]) -> Dict[str, Any]
    ├── # Produces a titled display image for a scene model, reusing a cached snapshot when available or rendering and caching otherwise, then overlaying cameras.
    ├── def _validate_inputs [local]
    │   ├── impls assert scene_model is a BaseSceneModel
    │   ├── impls assert camera is a Camera
    │   ├── impls assert resolution is None or a length-two tuple of positive int/scalar-tensor values
    │   ├── impls assert camera_name is None or a string
    │   ├── impls assert display_cameras is None or a list of Camera objects
    │   ├── impls assert title is None or a string
    │   └── impls assert device is None or a torch.device
    ├── calls _validate_inputs
    ├── def _normalize_inputs [local]
    │   ├── impls camera = camera.to(device if device is not None else scene_model.device)
    │   ├── if resolution is None
    │   │   └── impls resolution = camera.intrinsics.resolution  # the camera's own h and w are two of its params, so what raster to render is already answered
    │   ├── for each axis value
    │   │   └── impls convert int/scalar-tensor values to ints
    │   ├── if title is None
    │   │   └── impls title = ""
    │   ├── if device is None
    │   │   └── impls device = scene_model.device
    │   └── return camera, resolution, title, device
    ├── calls _normalize_inputs
    ├── if camera_name is not None
    │   └── impls image = scene_model._get_snapshot(camera_name)
    ├── if image is None
    │   ├── calls render_rgb_from_mesh_func(mesh=scene_model.model, camera=camera, resolution=resolution)
    │   └── if camera_name is not None
    │       └── impls scene_model._put_snapshot(camera_name, image.detach().cpu())
    ├── impls composed = BaseSceneModel._apply_camera_overlays(image=image, display_cameras=display_cameras, render_at_camera=camera, resolution=resolution)
    ├── impls display_payload = {"image": composed, "title": title}
    └── return display_payload
```

`models/three_d/meshes/render/uv_texture.py`

```text
uv_texture.py
├── from typing import Any, Tuple
├── import nvdiffrast.torch as dr
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
└── def render_uv_texture_aligned(renderer: Any, mesh: Mesh) -> Tuple[torch.Tensor, torch.Tensor]
    ├── # Renders a UV-textured mesh into the renderer's aligned image space via nvdiffrast, returning a mask and the RGB image.
    ├── impls read mesh.texture (a MeshTextureUVTextureMap) for verts_uvs and uv_texture_map  # impls-node-one-step:skip
    ├── impls mesh = mesh.to(uv_convention="top_left")
    └── if renderer.ctx is None
        ├── if renderer.use_opengl
        │   └── impls renderer.ctx = dr.RasterizeGLContext(device=device)
        └── else
            └── impls renderer.ctx = dr.RasterizeCudaContext(device=device)
```
