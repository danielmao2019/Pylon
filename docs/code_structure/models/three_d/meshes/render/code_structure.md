# `models/three_d/meshes/render/` code skeleton

## Code implementation structure

`models/three_d/meshes/render/__init__.py`

```text
__init__.py
├── from models.three_d.meshes.render.core import render_rgb_from_mesh
├── from models.three_d.meshes.render.display import render_display
└── from models.three_d.meshes.render.uv_texture import render_uv_texture_aligned
```

`models/three_d/meshes/render/core.py`

```text
core.py
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.mesh.convert import mesh_to_pytorch3d
├── from data.structures.three_d.mesh.mesh import Mesh
├── @torch.no_grad()
├── def render_rgb_from_mesh(mesh: Mesh, camera: Camera, resolution: Tuple[int, int], background: Tuple[int, int, int] = (0, 0, 0), return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders an RGB image (and optionally a validity mask) from a triangle Mesh using PyTorch3D.
│   ├── calls mesh_to_pytorch3d(mesh=mesh, device=device, dtype=torch.float32)
│   ├── calls _prepare_cameras(camera=camera, resolution=resolution, device=device)
│   ├── calls _build_rasterizer(cameras=cameras, resolution=resolution)
│   ├── calls _build_shader(cameras=cameras, device=device, background_color=background)
│   └── if return_mask
│       ├── impls fragments = rasterizer(meshes); valid_mask = fragments.pix_to_face[0, :, :, 0] >= 0
│       └── return rgb, valid_mask
├── def _prepare_cameras(camera: Camera, resolution: Tuple[int, int], device: torch.device) -> CamerasBase
│   ├── # Builds a PyTorch3D camera from a repo Camera, converting to PyTorch3D's right-handed convention and lifting zfar to float32 max, dispatched on the repo Camera's intrinsics model.
│   ├── if camera.intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   └── impls build a PerspectiveCameras from camera.intrinsics.matrix
│   ├── else
│   │   └── impls build an OrthographicCameras from the weak-perspective scale + principal-point intrinsics (no perspective divide)
│   └── return
├── def _build_rasterizer(cameras: CamerasBase, resolution: Tuple[int, int]) -> MeshRasterizer
│   └── # Builds a single-sample, no-blur MeshRasterizer for the given cameras and resolution.
└── def _build_shader(cameras: CamerasBase, device: torch.device, background_color: Tuple[int, int, int]) -> SoftPhongShader
    └── # Builds a flat-ambient SoftPhongShader with the given normalized background color.
```

`models/three_d/meshes/render/core_blender.py`

```text
core_blender.py
├── from data.structures.three_d.camera.camera import Camera
├── def render_rgb_from_mesh_blender(mesh_collection_name: str, camera: Camera, resolution: Tuple[int, int], background: Tuple[int, int, int] = (0, 0, 0), engine: str = 'CYCLES', device: str = 'GPU', view_layer_name: str = 'View Layer', return_mask: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
│   ├── # Renders RGB (and optional mask) of a named mesh collection using Blender's renderer, restoring scene state afterward.
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
│   ├── # Creates a Blender camera object from a repo Camera's intrinsics/extrinsics in the standard convention.
│   └── calls _torch_to_matrix_blender(extrinsics)
├── def _torch_to_matrix_blender(tensor: torch.Tensor) -> 'Matrix'
│   ├── # Converts a 4x4 torch transform tensor into a mathutils Matrix, raising if the shape is wrong.
│   └── if array.shape != (4, 4)
│       └── raise ValueError
├── def _configure_render_settings_blender(scene: 'bpy.types.Scene', resolution: Tuple[int, int]) -> Dict[str, Union[int, float, str]]
│   └── # Saves and overrides the scene render resolution/aspect/color-mode settings, returning the previous values.
├── def _configure_world_background_blender(scene: 'bpy.types.Scene', background: Tuple[int, int, int]) -> Dict[str, object]
│   ├── # Saves and overrides the scene world's flat background color, creating a world if none exists.
│   └── if prev_world is None
│       └── impls scene.world = bpy.data.worlds.new('mesh_world_blender'); created_world = True
├── def _configure_layer_mask_blender(view_layer: 'bpy.types.ViewLayer', enable_mask: bool) -> bool
│   └── # Saves and sets the view layer's object-index pass flag, returning the previous value.
├── def _configure_render_engine_blender(scene: 'bpy.types.Scene', engine: str, device: str) -> Tuple[str, str, str]
│   ├── # Saves and sets the render engine plus Cycles device/compute-type, returning the previous engine/device/compute.
│   ├── if engine_normalized == 'EEVEE'
│   │   └── impls engine_normalized = 'BLENDER_EEVEE'
│   └── if engine_normalized == 'CYCLES'
│       └── if cycles_addon
│           ├── if device.upper() == 'GPU'
│           │   └── for each candidate in ('OPTIX', 'CUDA', 'HIP', 'METAL')
│           │       └── if candidate in prefs.get_devices()
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
│   └── # Restores the previously saved scene render resolution/aspect/color-mode settings.
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
    └── # Restores the view layer's previous object-index pass flag.
```

`models/three_d/meshes/render/display.py`

```text
display.py
├── from data.structures.three_d.camera.camera import Camera
├── from models.three_d.base import BaseSceneModel
├── from models.three_d.meshes.render.core import render_rgb_from_mesh as render_rgb_from_mesh_func
└── def render_display(scene_model: BaseSceneModel, camera: Camera, resolution: Tuple[int, int], camera_name: Optional[str], display_cameras: Optional[List[Camera]], title: Optional[str], device: Optional[torch.device]) -> Dict[str, Any]
    ├── # Produces a titled display image for a scene model, reusing a cached snapshot when available or rendering and caching otherwise, then overlaying cameras.
    ├── if camera_name is not None
    │   └── impls image = scene_model._get_snapshot(camera_name)
    ├── if image is None
    │   ├── calls render_rgb_from_mesh_func(mesh=scene_model.model, camera=camera, resolution=resolution)
    │   └── if camera_name is not None
    │       └── impls scene_model._put_snapshot(camera_name, image.detach().cpu())
    └── impls composed = BaseSceneModel._apply_camera_overlays(image=image, display_cameras=display_cameras, render_at_camera=camera, resolution=resolution)
```

`models/three_d/meshes/render/uv_texture.py`

```text
uv_texture.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
└── def render_uv_texture_aligned(renderer: Any, mesh: Mesh) -> Tuple[torch.Tensor, torch.Tensor]
    ├── # Renders a UV-textured mesh into the renderer's aligned image space via nvdiffrast, returning a mask and the RGB image.
    ├── impls read mesh.texture (a MeshTextureUVTextureMap) for verts_uvs and uv_texture_map  # impls-node-one-step:skip
    ├── impls mesh = mesh.to(convention="top_left")
    └── if renderer.ctx is None
        ├── if renderer.use_opengl
        │   └── impls renderer.ctx = dr.RasterizeGLContext(device=device)
        └── else
            └── impls renderer.ctx = dr.RasterizeCudaContext(device=device)
```
