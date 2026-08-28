# `models/three_d/meshes/texture/` code skeleton

## 1. Code structure trees

`models/three_d/meshes/texture/extract/camera_geometry.py`

```text
camera_geometry.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from models.three_d.point_cloud.ops.world_to_camera_transform import world_to_camera_transform
├── PERSPECTIVE_DEPTH_FLOOR = 1e-8  # float; the smallest camera-space depth a perspective divide stays finite at, and the one only a perspective model needs
├── def render_camera_face_index_buffer(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Render a one-view camera-space face-index buffer.
│   └── calls _camera_verts_to_clip(verts_camera=verts_camera, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
├── def render_camera_depth_buffer(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Render a one-view camera-space depth buffer.
│   └── calls _camera_verts_to_clip(verts_camera=verts_camera, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
├── def project_verts_to_image(verts: torch.Tensor, camera: Cameras, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Project world-space verts to image pixels for one view.
│   ├── calls _verts_world_to_camera(verts=verts, camera=camera)
│   ├── calls camera_single.intrinsics.project(points_camera=verts_camera, inplace=False)  # the model's own projection, never one model's formula written out here
│   ├── calls compute_points_in_front_of_camera(points_camera=verts_camera, intrinsics=camera_single.intrinsics)
│   ├── impls valid = in-front points inside image bounds
│   └── return xy, verts_camera's z column, verts_camera and valid  # the depth stays the ordering key even under a model that projects without it
├── def compute_points_in_front_of_camera(points_camera: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor
│   ├── # Mark which camera-space points the camera's own model can see.
│   ├── if intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   └── return the points whose z passes PERSPECTIVE_DEPTH_FLOOR  # [N]; the half-space a perspective divide is defined on
│   ├── if intrinsics.model == "ortho"
│   │   └── return an all-true mask over points_camera's rows  # [N]; parallel rays reach the whole depth axis, leaving an ortho camera no behind
│   └── assert 0, "Should not reach here."
├── def compute_camera_view_directions(points_camera: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor
│   ├── # Compute each camera-space point's unit direction back toward the camera under the camera's own model.
│   ├── if intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   └── return points_camera negated and normalized per row  # [N, 3]; each pinhole ray runs back to the camera's own centre
│   ├── if intrinsics.model == "ortho"
│   │   └── return the negative z axis broadcast over points_camera's rows  # [N, 3]; the rays run parallel, so one axis stands for every one of them
│   └── assert 0, "Should not reach here."
├── def _camera_verts_to_clip(verts_camera: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Convert camera-space verts to clip-space for rasterization.
│   ├── impls z_camera = verts_camera's z column
│   ├── if intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   ├── impls z_camera = z_camera clamped to PERSPECTIVE_DEPTH_FLOOR  # the w a perspective divide would otherwise take to zero
│   │   └── impls w = z_camera
│   ├── if intrinsics.model == "ortho"
│   │   └── impls w = a ones column matching z_camera  # an ortho projection divides by no depth, so the homogeneous coordinate is unit
│   ├── impls verts_camera_floored = verts_camera with clamped z
│   ├── calls intrinsics.project(points_camera=verts_camera_floored, inplace=False)  # x_pixel and y_pixel off the model's own projection, its perspective divide or its plain scale
│   ├── impls x_ndc = twice x_pixel over the last column index, less one
│   ├── impls y_ndc = one less twice y_pixel over the last row index  # the raster's y-down origin onto clip space's y-up
│   ├── impls z_ndc = z_camera normalized over its own min-to-max range, carried onto [-1, 1]
│   ├── impls clip_verts = the (x_ndc, y_ndc, z_ndc, 1) columns each scaled by w, under one leading batch axis  # [1, V, 4]
│   └── return clip_verts
└── def _verts_world_to_camera(verts: torch.Tensor, camera: Cameras) -> torch.Tensor
    ├── # Transform one-view world-space verts to camera-space verts.
    └── calls world_to_camera_transform(points=verts, extrinsics=camera_single.extrinsics.extrinsics, inplace=False)
```

`models/three_d/meshes/texture/extract/extract.py`

```text
extract.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.texel_face_map import build_texel_face_map
├── from data.structures.three_d.mesh.texture.validate_vertex_color import validate_vertex_color
├── from models.three_d.meshes.texture.extract.camera_geometry import project_verts_to_image
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility import compute_f_visibility_mask
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import compute_f_visibility_mask_v2
├── from models.three_d.meshes.texture.extract.visibility.vertex_visibility import compute_v_visibility_mask
├── from models.three_d.meshes.texture.extract.weights.normal_weights import compute_f_normals_weights, compute_v_normals_weights
├── from models.three_d.meshes.texture.extract.weights.weights_cfg import normalize_weights_cfg, validate_weights_cfg
├── def extract_texture_from_images(mesh: Union[Mesh, List[Mesh]], images: Union[torch.Tensor, List[torch.Tensor]], cameras: Cameras, weights_cfg: Dict[str, Any]={}, texture_size: int=1024, default_color: float=0.7, return_valid_mask: bool=False, texel_visibility_method: str='v1', polygon_rast_method: str='v2') -> Union[torch.Tensor, Dict[str, torch.Tensor]]
│   ├── # Extract texture from multi-view RGB images and cameras named in any pair of frames, normalizing both camera halves and the UV origin on entry.
│   ├── calls validate_weights_cfg(weights_cfg=weights_cfg)
│   ├── calls normalize_weights_cfg(weights_cfg=weights_cfg)
│   ├── calls cameras.to(intr_convention='standard', extr_convention='opencv')  # normalize both halves: the pose onto opencv's camera-to-world, and the image plane onto the source frame's own top-left-origin pixels, which is what every fx / cx below is read as
│   ├── if not extract_uv_texture_map
│   │   └── calls _extract_vertex_color_from_images(meshes=meshes, images_nchw=images_nchw, cameras=cameras, weights_cfg=weights_cfg, default_color=default_color)
│   ├── for each view_mesh in meshes
│   │   └── calls view_mesh.to(uv_convention='obj')  # normalize the input UV origin to uv_convention='obj' (v=0 at bottom), the projector's expected UV convention
│   └── calls _extract_uv_texture_map_from_images(meshes=meshes, images_nchw=images_nchw, cameras=cameras, weights_cfg=weights_cfg, texture_size=texture_size, default_color=default_color, texel_visibility_method=texel_visibility_method, polygon_rast_method=polygon_rast_method)
├── def _extract_vertex_color_from_images(meshes: List[Mesh], images_nchw: torch.Tensor, cameras: Cameras, weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Fuse per-view projected vertex colors into one vertex-color tensor.
│   ├── for view_idx in range(images_nchw.shape[0])
│   │   └── calls _extract_vertex_color_from_single_image(mesh=meshes[view_idx], image=images_nchw[view_idx], camera=cameras[view_idx:view_idx + 1], weights_cfg=weights_cfg, default_color=default_color)
│   └── calls _fuse_vertex_color_observations(observations=observations, weights_cfg=weights_cfg, default_color=default_color)
├── def _fuse_vertex_color_observations(observations: List[Dict[str, torch.Tensor]], weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Fuse one-view vertex-color observations into one vertex-color tensor.
│   ├── if multi_view_robustness == 'none'
│   │   └── impls accumulate each observation's weighted texture into the running color numerator and weight denominator  # impls-node-one-step:skip
│   ├── else
│   │   └── calls validate_vertex_color(obj=provisional_vertex_color)
│   ├── calls validate_vertex_color(obj=vertex_color)
│   └── calls validate_vertex_color(obj=vertex_color)
├── def _extract_vertex_color_from_single_image(mesh: Mesh, image: torch.Tensor, camera: Cameras, weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Extract one-view vertex colors and corresponding per-vertex weights.
│   ├── calls compute_v_visibility_mask(mesh=mesh, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]))
│   ├── if weights == 'normals'
│   │   └── calls compute_v_normals_weights(mesh=mesh, camera=camera, weights_cfg=weights_cfg)
│   ├── else
│   │   └── impls vertex_weight = visibility_mask
│   └── calls _project_v_colors(mesh=mesh, image=image, camera=camera, default_color=default_color)
├── def _project_v_colors(mesh: Mesh, image: torch.Tensor, camera: Cameras, default_color: float) -> torch.Tensor
│   ├── # Project one image to verts and sample per-vertex RGB colors.
│   └── calls project_verts_to_image(verts=mesh.verts, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]))
├── def _extract_uv_texture_map_from_images(meshes: List[Mesh], images_nchw: torch.Tensor, cameras: Cameras, weights_cfg: Dict[str, Any], texture_size: int, default_color: float, texel_visibility_method: str, polygon_rast_method: str='v2') -> Dict[str, torch.Tensor]
│   ├── # Fuse per-view UV observations into one UV texture map.
│   ├── calls build_texel_face_map(mesh=reference_mesh, texture_size=texture_size)
│   ├── for view_idx in range(images_nchw.shape[0])
│   │   └── calls _extract_uv_texture_map_from_single_image(mesh=meshes[view_idx], image=images_nchw[view_idx], camera=cameras[view_idx:view_idx + 1], weights_cfg=weights_cfg, texel_face_map=texel_face_map, texel_visibility_method=texel_visibility_method, polygon_rast_method=polygon_rast_method)
│   └── calls _fuse_uv_texture_observations(observations=observations, weights_cfg=weights_cfg, default_color=default_color)
├── def _fuse_uv_texture_observations(observations: List[Dict[str, torch.Tensor]], weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Fuse one-view UV observations into one UV texture map.
│   ├── if multi_view_robustness == 'none'
│   │   └── impls accumulate each observation's weighted texture into the running uv numerator and weight denominator  # impls-node-one-step:skip
│   ├── else
│   │   └── calls _validate_rgb_image(obj=provisional_uv_texture_map)
│   ├── calls _validate_rgb_image(obj=uv_texture_map)
│   └── calls _validate_rgb_image(obj=uv_texture_map)
├── def _extract_uv_texture_map_from_single_image(mesh: Mesh, image: torch.Tensor, camera: Cameras, weights_cfg: Dict[str, Any], texel_face_map: Dict[str, torch.Tensor], texel_visibility_method: str='v1', polygon_rast_method: str='v2') -> Dict[str, torch.Tensor]
│   ├── # Extract one-view UV texture observation and UV weight map, both keyed by the mesh's uv_convention='obj' UV layout.
│   ├── if texel_visibility_method == 'v1'
│   │   └── calls compute_f_visibility_mask(verts=mesh.verts, faces=mesh.faces, face_verts_uvs=mesh.texture.verts_uvs[mesh.texture.faces_uvs], camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]), texel_face_map=texel_face_map, polygon_rast_method=polygon_rast_method)
│   ├── else
│   │   └── calls compute_f_visibility_mask_v2(verts=mesh.verts, faces=mesh.faces, face_verts_uvs=mesh.texture.verts_uvs[mesh.texture.faces_uvs], camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]), texel_face_map=texel_face_map)
│   ├── if weights == 'normals'
│   │   ├── calls compute_f_normals_weights(mesh=mesh, camera=camera, weights_cfg=weights_cfg)
│   │   └── calls _rasterize_face_weights_to_uv(face_weight=face_normals_weight, texel_face_map=texel_face_map)
│   ├── else
│   │   └── impls uv_weight = uv_visibility_mask
│   └── calls _project_f_colors(mesh=mesh, image=image, camera=camera, texel_face_map=texel_face_map)
├── def _project_f_colors(mesh: Mesh, image: torch.Tensor, camera: Cameras, texel_face_map: Dict[str, torch.Tensor]) -> torch.Tensor
│   ├── # Project one image into UV space using rasterized UV correspondence.
│   ├── calls project_verts_to_image(verts=mesh.verts, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]))
│   ├── def _interpolate_uv_texel_image_coords(projected_vertex_xy: torch.Tensor, texel_face_map: Dict[str, torch.Tensor]) -> torch.Tensor [local]
│   │   ├── # Interpolate image-space coordinates for every occupied UV texel.
│   │   ├── impls texel_face_index = texel_face_map["texel_face_index"]
│   │   ├── impls texel_face_barycentric = texel_face_map["texel_face_barycentric"]
│   │   ├── impls per_corner_xy = projected_vertex_xy at mesh.faces of texel_face_index clamped to zero
│   │   ├── impls interpolated_xy = per_corner_xy weighted by texel_face_barycentric, summed over the three corners
│   │   ├── impls zero interpolated_xy wherever texel_face_index is the unoccupied sentinel
│   │   └── return interpolated_xy under one leading batch axis  # [1, T, T, 2]
│   ├── calls _interpolate_uv_texel_image_coords(projected_vertex_xy=xy, texel_face_map=texel_face_map)
│   ├── def _sample_uv_texel_colors_from_source_image(interpolated_uv_xy: torch.Tensor, image: torch.Tensor) -> torch.Tensor [local]
│   │   ├── # Sample source-image colors at interpolated UV texel image coordinates.
│   │   ├── impls grid_x = interpolated_uv_xy's x over the image's last column index, carried onto [-1, 1]
│   │   ├── impls grid_y = interpolated_uv_xy's y over the image's last row index, carried onto [-1, 1]
│   │   ├── impls sampling_grid = grid_x stacked with grid_y on a trailing axis
│   │   ├── calls F.grid_sample(input=image under one leading batch axis, grid=sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
│   │   └── return the sampled image permuted to NHWC  # [1, T, T, 3]
│   ├── calls _sample_uv_texel_colors_from_source_image(interpolated_uv_xy=interpolated_uv_xy, image=image)
│   └── calls _validate_rgb_image(obj=uv_texture)
├── def _validate_rgb_image(obj: Any) -> None
│   ├── # Validate that an object is an RGB image tensor (CHW/HWC/NCHW/NHWC, uint8 [0,255] or float32 [0,1]).
│   ├── assert obj is a torch.Tensor
│   ├── assert obj is rank 3 or rank 4
│   ├── if obj is rank 3
│   │   ├── assert its first or its last axis is the RGB triple  # CHW or HWC
│   │   └── impls image_height, image_width = its two remaining axes, in that layout's order
│   ├── else
│   │   ├── assert its batch axis is one
│   │   ├── assert its second or its last axis is the RGB triple  # NCHW or NHWC
│   │   └── impls image_height, image_width = its two remaining axes, in that layout's order
│   ├── assert the resolution is positive on both axes
│   ├── if obj's dtype is torch.uint8
│   │   └── return  # uint8 carries the [0, 255] range in the dtype itself
│   ├── assert obj's dtype is torch.float32
│   ├── assert every obj entry is finite
│   ├── assert obj's smallest entry is at least 0.0
│   └── assert obj's largest entry is at most 1.0
└── def _rasterize_face_weights_to_uv(face_weight: torch.Tensor, texel_face_map: Dict[str, torch.Tensor]) -> torch.Tensor
    ├── # Map per-face weights to per-UV-pixel weights for one view.
    ├── impls texel_face_index = texel_face_map["texel_face_index"]
    ├── impls occupied_mask = texel_face_index at or above zero
    ├── impls gathered = face_weight indexed by texel_face_index clamped to zero  # the -1 sentinel would index out of range
    ├── impls uv_weight = gathered where occupied_mask holds, zero elsewhere
    └── return uv_weight clamped to a zero floor, under one leading batch axis and one trailing channel axis  # [1, T, T, 1]
```

`models/three_d/meshes/texture/extract/weights/normal_weights.py`

```text
normal_weights.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.mesh.mesh import Mesh
├── from models.three_d.meshes.ops.normals import compute_vertex_normals
├── from models.three_d.meshes.texture.extract.camera_geometry import compute_camera_view_directions, _verts_world_to_camera
├── def compute_v_normals_weights(mesh: Mesh, camera: Cameras, weights_cfg: Dict[str, Any]) -> torch.Tensor
│   ├── # Compute one-view per-vertex normal-alignment weights.
│   ├── impls normals_weight_power, normals_weight_threshold = the normalized weights_cfg's two entries
│   ├── calls _verts_world_to_camera(verts=mesh.verts, camera=camera)
│   ├── calls compute_vertex_normals(verts=verts_camera, faces=mesh.faces, weights="area")
│   ├── impls normals_camera = the vertex normals on the mesh's device as float32
│   ├── assert every normals_camera row is unit length to within 1e-5
│   ├── calls compute_camera_view_directions(points_camera=verts_camera, intrinsics=camera[0].intrinsics)
│   ├── impls alignment = the per-row dot of normals_camera with view_direction, clamped to [0, 1]
│   ├── assert every alignment lies within [0, 1]
│   ├── impls alignment = alignment zeroed wherever it falls below normals_weight_threshold
│   ├── impls alignment = alignment raised to normals_weight_power
│   └── return alignment  # [V]
└── def compute_f_normals_weights(mesh: Mesh, camera: Cameras, weights_cfg: Dict[str, Any]) -> torch.Tensor
    ├── # Compute one-view per-face normal-alignment weights.
    ├── impls normals_weight_power, normals_weight_threshold = the normalized weights_cfg's two entries
    ├── calls _verts_world_to_camera(verts=mesh.verts, camera=camera)
    ├── impls v0_camera, v1_camera, v2_camera = verts_camera gathered at each face's three corners
    ├── impls face_normals_camera = the cross product of (v1_camera - v0_camera) with (v2_camera - v0_camera)
    ├── assert every face normal carries a non-zero magnitude
    ├── impls face_normals_camera = face_normals_camera over its own row norms
    ├── impls face_centers_camera = per-face corner means
    ├── calls compute_camera_view_directions(points_camera=face_centers_camera, intrinsics=camera[0].intrinsics)
    ├── impls alignment = the per-row dot of face_normals_camera with face_view_direction, clamped to [0, 1]
    ├── assert every alignment lies within [0, 1]
    ├── impls alignment = alignment zeroed wherever it falls below normals_weight_threshold
    ├── impls alignment = alignment raised to normals_weight_power
    └── return alignment  # [F]
```

`models/three_d/meshes/texture/extract/weights/weights_cfg.py`

```text
weights_cfg.py
├── WEIGHTS_CFG_ALLOWED_KEYS
├── def validate_weights_cfg(weights_cfg: Dict[str, Any]) -> None
│   ├── # Validate one texture-extraction weights config.
│   ├── assert weights_cfg is a dict
│   ├── assert weights_cfg's keys stay within WEIGHTS_CFG_ALLOWED_KEYS
│   ├── assert its weights, when present, is "visible" or "normals"
│   ├── assert its normals_weight_power, when present, is a float above 0.0
│   ├── assert its normals_weight_threshold, when present, is a float within [0.0, 1.0]
│   ├── assert its multi_view_robustness, when present, is "none" or "residual_gaussian"
│   ├── assert its robustness_tau, when present, is a float above 0.0
│   └── assert its first_frame_blending_weight_power, when present, is a float above 0.0
└── def normalize_weights_cfg(weights_cfg: Dict[str, Any], default_weights: str) -> Dict[str, Any]
    ├── # Normalize one texture-extraction weights config.
    ├── impls weights_cfg = a copy of weights_cfg
    ├── if weights_cfg carries no weights
    │   └── impls weights_cfg["weights"] = default_weights
    ├── if weights_cfg carries no normals_weight_power
    │   └── impls weights_cfg["normals_weight_power"] = 1.0
    ├── if weights_cfg carries no normals_weight_threshold
    │   └── impls weights_cfg["normals_weight_threshold"] = 0.0
    ├── if weights_cfg carries no multi_view_robustness
    │   └── impls weights_cfg["multi_view_robustness"] = "none"
    ├── if weights_cfg carries no robustness_tau
    │   └── impls weights_cfg["robustness_tau"] = 0.2
    ├── if weights_cfg carries no first_frame_blending_weight_power
    │   └── impls weights_cfg["first_frame_blending_weight_power"] = 2.0
    └── return weights_cfg
```

`models/three_d/meshes/texture/extract/visibility/texel_visibility.py`

```text
texel_visibility.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from models.three_d.meshes.texture.extract.camera_geometry import _verts_world_to_camera
├── from models.three_d.meshes.texture.extract.weights.normal_weights import compute_f_normals_weights
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility_geometry import build_uv_polygon_texel_intersections, build_uv_triangle_texel_intersections_v2, build_visible_face_pixel_polygons, camera_verts_to_pixel, clip_convex_polygons_to_pixel_squares, _compute_convex_polygon_areas, compute_face_depth_ordering_coefficients, duplicate_wrapped_uv_polygons, project_screen_polygons_to_face_uv, triangulate_convex_uv_polygons
├── def compute_f_visibility_mask(verts: torch.Tensor, faces: torch.Tensor, face_verts_uvs: torch.Tensor, camera: Cameras, image_height: int, image_width: int, texel_face_map: Dict[str, torch.Tensor], polygon_rast_method: str='v2') -> torch.Tensor
│   ├── # Compute one-view UV-pixel visibility mask from exact camera-pixel footprints.
│   ├── calls _verts_world_to_camera(verts=verts, camera=camera)
│   ├── calls compute_f_normals_weights(mesh=Mesh(verts=verts, faces=faces), camera=camera, weights_cfg={'weights': 'normals'})
│   ├── calls _compute_visible_uv_polygon_regions_from_camera_pixels(verts_camera=verts_camera, faces=faces, intrinsics=camera[0].intrinsics, image_height=image_height, image_width=image_width, face_front_facing_mask=face_front_facing_mask, camera_face_verts_uvs=face_verts_uvs)
│   └── calls _compute_visible_uv_texels_from_uv_polygon_regions(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts, texture_size=int(texel_face_map['texel_face_index'].shape[0]), polygon_rast_method=polygon_rast_method)
├── def _compute_visible_uv_polygon_regions_from_camera_pixels(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int, face_front_facing_mask: torch.Tensor, camera_face_verts_uvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Compute exact visible UV polygon regions from camera pixels.
│   ├── calls camera_verts_to_pixel(verts_camera=verts_camera, intrinsics=intrinsics)
│   ├── calls _compute_visible_screen_space_polygon_regions_inside_camera_pixels(face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
│   └── calls _map_visible_screen_space_polygon_regions_to_uv(visible_screen_polygon_verts=visible_screen_polygon_verts, visible_screen_polygon_vertex_counts=visible_screen_polygon_vertex_counts, visible_screen_polygon_face_indices=visible_screen_polygon_face_indices, face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, face_verts_uvs=face_verts_uvs, intrinsics=intrinsics)
├── def _compute_visible_screen_space_polygon_regions_inside_camera_pixels(face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute exact visible screen-space polygon regions inside each camera pixel.
│   ├── calls _compute_face_pixel_polygon_intersections_without_occlusion(face_screen_verts=face_screen_verts, image_height=image_height, image_width=image_width)
│   └── calls _compute_visible_screen_space_polygon_regions_with_occlusion(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, clipped_pixel_indices=clipped_pixel_indices, clipped_face_indices=clipped_face_indices, face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
├── def _compute_face_pixel_polygon_intersections_without_occlusion(face_screen_verts: torch.Tensor, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute all face-pixel polygon intersections without considering occlusion.
│   ├── def _compute_projected_face_pixel_bounds() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] [local]
│   │   ├── # Compute candidate pixel bounds for each projected face.
│   │   ├── impls face_x_min, face_x_max = the per-face min and max over its screen verts' x  # impls-node-one-step:skip
│   │   ├── impls face_y_min, face_y_max = the per-face min and max over its screen verts' y  # impls-node-one-step:skip
│   │   ├── impls pixel_x_start = ceil(face_x_min - 0.5), as long                             # the first pixel whose own square the face can reach
│   │   ├── impls pixel_x_end = floor(face_x_max + 0.5), as long
│   │   ├── impls pixel_y_start = ceil(face_y_min - 0.5), as long
│   │   ├── impls pixel_y_end = floor(face_y_max + 0.5), as long
│   │   ├── impls clamp each start and end into its own image axis  # impls-node-one-step:skip
│   │   ├── impls pixel_x_count = pixel_x_end - pixel_x_start + 1, floored at zero
│   │   ├── impls pixel_y_count = pixel_y_end - pixel_y_start + 1, floored at zero
│   │   ├── impls pair_count_per_face = pixel_x_count * pixel_y_count
│   │   ├── impls total_pair_count = the sum of pair_count_per_face, as an int
│   │   └── return pixel_x_start, pixel_y_start, pixel_x_count, pixel_y_count, pixel_x_end, pixel_y_end, pair_count_per_face, total_pair_count
│   ├── calls _compute_projected_face_pixel_bounds()
│   ├── def _enumerate_candidate_face_pixel_pairs(pair_count_per_face: torch.Tensor, pixel_x_start: torch.Tensor, pixel_y_start: torch.Tensor, pixel_x_count: torch.Tensor, total_pair_count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   ├── # Enumerate all candidate face-pixel pairs.
│   │   ├── impls repeated_face_indices = each face index repeated pair_count_per_face times
│   │   ├── impls pair_start_offsets = the exclusive prefix sum of pair_count_per_face
│   │   ├── impls pair_offsets = each pair's global index less its own face's pair_start_offset
│   │   ├── impls local_pixel_y_offset = pair_offsets floor-divided by that face's pixel_x_count
│   │   ├── impls local_pixel_x_offset = pair_offsets modulo that face's pixel_x_count
│   │   ├── impls pixel_x = that face's pixel_x_start plus local_pixel_x_offset
│   │   ├── impls pixel_y = that face's pixel_y_start plus local_pixel_y_offset
│   │   └── return repeated_face_indices, pixel_x, pixel_y  # [total_pair_count] each
│   ├── calls _enumerate_candidate_face_pixel_pairs(pair_count_per_face=pair_count_per_face, pixel_x_start=pixel_x_start, pixel_y_start=pixel_y_start, pixel_x_count=pixel_x_count, total_pair_count=total_pair_count)
│   ├── def _clip_face_triangles_to_pixel_squares(repeated_face_indices: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor, total_pair_count: int) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Clip projected face triangles to candidate pixel squares.
│   │   └── calls clip_convex_polygons_to_pixel_squares(polygon_verts=polygon_verts, polygon_vertex_counts=polygon_vertex_counts, pixel_x=pixel_x.to(dtype=torch.float32), pixel_y=pixel_y.to(dtype=torch.float32))
│   ├── calls _clip_face_triangles_to_pixel_squares(repeated_face_indices=repeated_face_indices, pixel_x=pixel_x, pixel_y=pixel_y, total_pair_count=total_pair_count)
│   ├── def _pack_valid_face_pixel_polygons(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor, repeated_face_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   ├── # Reject degenerate overlaps and pack the surviving polygons.
│   │   └── calls _compute_convex_polygon_areas(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts)
│   └── calls _pack_valid_face_pixel_polygons(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, pixel_x=pixel_x, pixel_y=pixel_y, repeated_face_indices=repeated_face_indices)
├── def _compute_visible_screen_space_polygon_regions_with_occlusion(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, clipped_pixel_indices: torch.Tensor, clipped_face_indices: torch.Tensor, face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute inter-polygon occlusion and remove hidden screen-space regions.
│   ├── def _compute_projected_face_depth_ordering_coefficients() -> torch.Tensor [local]
│   │   ├── # Compute the affine depth-ordering plane the projected faces are compared on.
│   │   └── calls compute_face_depth_ordering_coefficients(face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, intrinsics=intrinsics)
│   ├── calls _compute_projected_face_depth_ordering_coefficients()
│   ├── def _build_exact_visible_face_pixel_polygons(face_depth_ordering_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Resolve exact visible face-pixel polygons from clipped overlaps.
│   │   └── calls build_visible_face_pixel_polygons(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, clipped_pixel_indices=clipped_pixel_indices, clipped_face_indices=clipped_face_indices, face_depth_ordering_coefficients=face_depth_ordering_coefficients)
│   ├── calls _build_exact_visible_face_pixel_polygons(face_depth_ordering_coefficients=face_depth_ordering_coefficients)
│   ├── def _pack_visible_polygon_outputs(visible_polygon_verts: torch.Tensor, visible_polygon_vertex_counts: torch.Tensor, visible_polygon_face_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   ├── # Pack exact visible polygons into the downstream tensor format.
│   │   └── return visible_polygon_verts, visible_polygon_vertex_counts and visible_polygon_face_indices, each contiguous
│   └── calls _pack_visible_polygon_outputs(visible_polygon_verts=visible_polygon_verts, visible_polygon_vertex_counts=visible_polygon_vertex_counts, visible_polygon_face_indices=visible_polygon_face_indices)
├── def _map_visible_screen_space_polygon_regions_to_uv(visible_screen_polygon_verts: torch.Tensor, visible_screen_polygon_vertex_counts: torch.Tensor, visible_screen_polygon_face_indices: torch.Tensor, face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, face_verts_uvs: torch.Tensor, intrinsics: CameraIntrinsics) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Map visible screen-space polygon regions into UV.
│   ├── def _gather_visible_polygon_face_geometry() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   ├── # Gather owning-face geometry for each visible polygon.
│   │   ├── impls polygon_face_screen_verts = face_screen_verts gathered at visible_screen_polygon_face_indices, contiguous
│   │   ├── impls polygon_face_vertex_depth = face_vertex_depth gathered at visible_screen_polygon_face_indices, contiguous
│   │   ├── impls polygon_face_verts_uvs = face_verts_uvs gathered at visible_screen_polygon_face_indices, contiguous
│   │   └── return polygon_face_screen_verts, polygon_face_vertex_depth, polygon_face_verts_uvs
│   ├── calls _gather_visible_polygon_face_geometry()
│   ├── def _project_screen_polygon_verts_to_uv(polygon_face_screen_verts: torch.Tensor, polygon_face_vertex_depth: torch.Tensor, polygon_face_verts_uvs: torch.Tensor) -> torch.Tensor [local]
│   │   ├── # Project visible screen polygons into UV.
│   │   └── calls project_screen_polygons_to_face_uv(polygon_verts=visible_screen_polygon_verts, face_screen_verts=polygon_face_screen_verts, face_vertex_depth=polygon_face_vertex_depth, face_verts_uvs=polygon_face_verts_uvs, intrinsics=intrinsics)
│   ├── calls _project_screen_polygon_verts_to_uv(polygon_face_screen_verts=polygon_face_screen_verts, polygon_face_vertex_depth=polygon_face_vertex_depth, polygon_face_verts_uvs=polygon_face_verts_uvs)
│   ├── def _pack_visible_uv_polygons(uv_polygon_verts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Pack UV polygons with their original vertex counts.
│   │   ├── impls packed_uv_polygon_verts = uv_polygon_verts contiguous
│   │   ├── impls packed_uv_polygon_vertex_counts = visible_screen_polygon_vertex_counts contiguous
│   │   └── return packed_uv_polygon_verts, packed_uv_polygon_vertex_counts  # the UV projection moves each vertex, never their number
│   └── calls _pack_visible_uv_polygons(uv_polygon_verts=uv_polygon_verts)
├── def _compute_visible_uv_texels_from_uv_polygon_regions(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor, texture_size: int, polygon_rast_method: str='v2') -> torch.Tensor
│   ├── # Compute visible UV texels from the UV polygon regions.
│   ├── if polygon_rast_method == 'v1'
│   │   └── calls _compute_uv_polygon_texel_contributions_v1(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts, texture_size=texture_size)
│   └── else
│       └── calls _compute_uv_polygon_texel_contributions_v2(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts, texture_size=texture_size)
├── def _compute_uv_polygon_texel_contributions_v1(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor, texture_size: int) -> torch.Tensor
│   ├── # Construct exact step-2 `v1` texel contributions for visible UV polygons.
│   ├── def _duplicate_wrap_crossing_polygons() -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Duplicate wrap-crossing polygons so the cylindrical UV union is preserved.
│   │   └── calls duplicate_wrapped_uv_polygons(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts)
│   ├── calls _duplicate_wrap_crossing_polygons()
│   └── calls build_uv_polygon_texel_intersections(uv_polygon_verts=wrapped_uv_polygon_verts, uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts, texture_size=texture_size)
└── def _compute_uv_polygon_texel_contributions_v2(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor, texture_size: int) -> torch.Tensor
    ├── # Construct approximate step-2 `v2` texel contributions for visible UV polygons.
    ├── def _duplicate_wrap_crossing_polygons() -> Tuple[torch.Tensor, torch.Tensor] [local]
    │   ├── # Duplicate wrap-crossing polygons so the cylindrical UV union is preserved.
    │   └── calls duplicate_wrapped_uv_polygons(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts)
    ├── calls _duplicate_wrap_crossing_polygons()
    ├── def _triangulate_wrapped_uv_polygons(wrapped_uv_polygon_verts: torch.Tensor, wrapped_uv_polygon_vertex_counts: torch.Tensor) -> torch.Tensor [local]
    │   ├── # Triangulate wrapped convex UV polygons into a triangle soup.
    │   └── calls triangulate_convex_uv_polygons(polygon_verts=wrapped_uv_polygon_verts, polygon_vertex_counts=wrapped_uv_polygon_vertex_counts)
    ├── calls _triangulate_wrapped_uv_polygons(wrapped_uv_polygon_verts=wrapped_uv_polygon_verts, wrapped_uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts)
    └── calls build_uv_triangle_texel_intersections_v2(uv_triangles=wrapped_uv_triangles, texture_size=texture_size)
```

`models/three_d/meshes/texture/extract/visibility/texel_visibility_geometry.py`

```text
texel_visibility_geometry.py
├── import torch
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── TARGET_MULTI_FACE_PIXEL_SPLIT_LINE_BUDGET = 2 ** 18  # int — the padded split-line rows one chunk may materialize at once, the memory bound _plan_multi_face_pixel_chunks plans against
├── def build_visible_face_pixel_polygons(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, clipped_pixel_indices: torch.Tensor, clipped_face_indices: torch.Tensor, face_depth_ordering_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Build exact visible face-pixel polygons in batched tensor form.
│   ├── calls _pack_face_pixel_polygons_by_pixel(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, clipped_pixel_indices=clipped_pixel_indices, clipped_face_indices=clipped_face_indices, face_depth_ordering_coefficients=face_depth_ordering_coefficients)
│   ├── if torch.any(single_face_pixel_mask)
│   │   └── calls _gather_visible_pixel_face_polygons(pixel_polygon_verts=pixel_polygon_verts[single_face_pixel_mask], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[single_face_pixel_mask], pixel_face_indices=pixel_face_indices[single_face_pixel_mask], pixel_face_slot_mask=pixel_face_valid_mask[single_face_pixel_mask])
│   └── calls _build_visible_multi_face_pixel_polygons(pixel_indices=pixel_indices[multi_face_pixel_mask], pixel_polygon_verts=pixel_polygon_verts[multi_face_pixel_mask], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[multi_face_pixel_mask], pixel_face_indices=pixel_face_indices[multi_face_pixel_mask], pixel_face_valid_mask=pixel_face_valid_mask[multi_face_pixel_mask], pixel_depth_ordering_coefficients=pixel_depth_ordering_coefficients[multi_face_pixel_mask])
├── def _build_visible_multi_face_pixel_polygons(pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_indices: torch.Tensor, pixel_face_valid_mask: torch.Tensor, pixel_depth_ordering_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Resolve visible polygons for the multi-face pixels in memory-bounded batches.
│   ├── calls _compute_multi_face_pixel_second_bucket_mask(pixel_polygon_verts=pixel_polygon_verts, pixel_polygon_vertex_counts=pixel_polygon_vertex_counts, pixel_face_valid_mask=pixel_face_valid_mask)
│   ├── if torch.any(first_bucket_mask)
│   │   └── calls _gather_visible_pixel_face_polygons(pixel_polygon_verts=pixel_polygon_verts[first_bucket_mask], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[first_bucket_mask], pixel_face_indices=pixel_face_indices[first_bucket_mask], pixel_face_slot_mask=pixel_face_valid_mask[first_bucket_mask])
│   ├── calls _plan_multi_face_pixel_chunks(face_count_per_pixel=face_count_per_pixel, max_verts_per_polygon=max_verts_per_polygon, target_split_line_budget=TARGET_MULTI_FACE_PIXEL_SPLIT_LINE_BUDGET)
│   └── for (chunk_start, chunk_end) in chunk_ranges
│       ├── calls _build_padded_pixel_split_line_coefficients(pixel_indices=pixel_indices[chunk_start:chunk_end], pixel_polygon_verts=pixel_polygon_verts[chunk_start:chunk_end], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[chunk_start:chunk_end], pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end])
│       ├── calls _build_batched_pixel_cell_polygons(pixel_indices=pixel_indices[chunk_start:chunk_end], pixel_polygon_verts=pixel_polygon_verts[chunk_start:chunk_end], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[chunk_start:chunk_end], pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end], pixel_split_line_coefficients=pixel_split_line_coefficients, pixel_split_line_valid_mask=pixel_split_line_valid_mask)
│       └── calls _assign_visible_faces_to_cells(cell_polygon_verts=cell_polygon_verts, cell_polygon_vertex_counts=cell_polygon_vertex_counts, cell_pixel_indices=cell_pixel_indices, pixel_polygon_verts=pixel_polygon_verts[chunk_start:chunk_end], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[chunk_start:chunk_end], pixel_face_indices=pixel_face_indices[chunk_start:chunk_end], pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end], pixel_depth_ordering_coefficients=pixel_depth_ordering_coefficients[chunk_start:chunk_end])
├── def _plan_multi_face_pixel_chunks(face_count_per_pixel: torch.Tensor, max_verts_per_polygon: int, target_split_line_budget: int) -> List[Tuple[int, int]]
│   ├── # Plan sorted multi-face pixel chunks under a split-line budget.
│   ├── if face_count_per_pixel is empty
│   │   └── return an empty list
│   ├── impls chunk_ranges = an empty list
│   ├── impls chunk_start = 0
│   ├── while chunk_start is below the pixel count
│   │   ├── impls chunk_end = chunk_start + 1
│   │   ├── while chunk_end is below the pixel count
│   │   │   ├── impls next_max_face_count = face_count_per_pixel at chunk_end
│   │   │   ├── impls prospective_chunk_size = chunk_end - chunk_start + 1  # the chunk this pixel would join
│   │   │   ├── impls prospective_split_line_count = next_max_face_count times max_verts_per_polygon, plus one split line per face pair  # impls-node-one-step:skip
│   │   │   ├── if prospective_chunk_size times prospective_split_line_count passes target_split_line_budget
│   │   │   │   └── break
│   │   │   └── impls chunk_end += 1
│   │   ├── impls append (chunk_start, chunk_end) to chunk_ranges
│   │   └── impls chunk_start = chunk_end
│   └── return chunk_ranges
├── def _gather_visible_pixel_face_polygons(pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_indices: torch.Tensor, pixel_face_slot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Gather selected pixel-face polygons into flat visible outputs.
│   └── return pixel_polygon_verts, pixel_polygon_vertex_counts and pixel_face_indices, each masked by pixel_face_slot_mask and contiguous
├── def _compute_multi_face_pixel_second_bucket_mask(pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_valid_mask: torch.Tensor) -> torch.Tensor
│   ├── # Detect which multi-face pixels require full overlap resolution.
│   └── calls _compute_pair_positive_area_overlap_mask(first_polygon_verts=first_pair_polygon_verts, first_polygon_vertex_counts=first_pair_polygon_vertex_counts, second_polygon_verts=second_pair_polygon_verts, second_polygon_vertex_counts=second_pair_polygon_vertex_counts)
├── def _compute_pair_positive_area_overlap_mask(first_polygon_verts: torch.Tensor, first_polygon_vertex_counts: torch.Tensor, second_polygon_verts: torch.Tensor, second_polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   ├── # Detect positive-area overlap for convex polygon pairs.
│   ├── if the pair count is zero
│   │   └── return an empty bool tensor
│   ├── impls first_vertex_valid_mask = the vertex slots below the first polygon's own count
│   ├── impls second_vertex_valid_mask = the vertex slots below the second polygon's own count
│   ├── impls first_min_x, first_max_x, first_min_y, first_max_y = the first polygon's valid-vertex bounds, its inactive slots held at ±inf
│   ├── impls second_min_x, second_max_x, second_min_y, second_max_y = the second polygon's valid-vertex bounds, its inactive slots held at ±inf
│   ├── impls bbox_overlap_mask = the pairs whose two bounding boxes strictly overlap on both axes
│   ├── if no pair passes bbox_overlap_mask
│   │   └── return an all-false mask
│   ├── impls first_edge_axes = the perpendicular of each valid first-polygon edge
│   ├── impls second_edge_axes = the perpendicular of each valid second-polygon edge
│   ├── impls candidate_axes = first_edge_axes concatenated with second_edge_axes  # the separating-axis candidates
│   ├── impls candidate_axis_valid_mask = the valid edges whose axis carries a non-degenerate norm
│   ├── impls first_projection_min, first_projection_max = the first polygon's valid verts projected onto each candidate axis
│   ├── impls second_projection_min, second_projection_max = the second polygon's valid verts projected onto each candidate axis
│   ├── impls separating_axis_mask = the valid candidate axes whose two projected intervals stay apart within 1.0e-12
│   ├── impls overlap_mask = the bbox-overlapping pairs carrying no separating axis, made contiguous
│   └── return overlap_mask  # one bool per pair
├── def _build_padded_pixel_split_line_coefficients(pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Build padded polygon-edge split-line tensors for all pixels.
│   └── calls _deduplicate_padded_pixel_split_lines(pixel_split_line_coefficients=edge_line_coefficients, pixel_split_line_valid_mask=edge_valid_mask)
├── def _deduplicate_padded_pixel_split_lines(pixel_split_line_coefficients: torch.Tensor, pixel_split_line_valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Deduplicate canonical split lines independently within each pixel.
│   ├── if no split line is valid
│   │   ├── impls deduplicated_coefficients = zeroed coefficients
│   │   ├── impls deduplicated_valid_mask = a zeroed valid mask
│   │   └── return deduplicated_coefficients, deduplicated_valid_mask
│   ├── assert the pixel count stays below float32's exact-integer range  # the pixel index rides as a float column below
│   ├── impls flat_pixel_indices = each valid line's own pixel row index
│   ├── impls flat_line_coefficients = the valid lines' coefficients
│   ├── assert every valid line carries a non-degenerate direction norm
│   ├── impls canonical_line_coefficients = flat_line_coefficients over that norm
│   ├── impls flip canonical_line_coefficients wherever its leading non-zero component is negative  # one representative per undirected line
│   ├── impls pixel_scoped_line_rows = flat_pixel_indices prefixed onto canonical_line_coefficients
│   ├── impls unique_pixel_scoped_line_rows = the distinct pixel_scoped_line_rows  # dedup lands per pixel because the pixel index is part of the row
│   ├── impls pixel_group_counts = the distinct line count per pixel
│   ├── impls within_group_indices = each unique line's rank inside its own pixel's group
│   ├── impls scatter unique_pixel_scoped_line_rows' coefficient columns into a zeroed padded tensor at its own (pixel, within_group_index) slot
│   ├── impls set the matching padded valid mask entries true
│   ├── impls deduplicated_coefficients = the padded coefficient tensor contiguous
│   ├── impls deduplicated_valid_mask = the padded valid mask contiguous
│   └── return deduplicated_coefficients, deduplicated_valid_mask
├── def _build_batched_pixel_cell_polygons(pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_valid_mask: torch.Tensor, pixel_split_line_coefficients: torch.Tensor, pixel_split_line_valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Build exact arrangement cells for all pixels in batched tensor form.
│   ├── for split_line_index in range(pixel_split_line_coefficients.shape[1])
│   │   ├── calls _clip_convex_polygons_to_half_plane(polygon_verts=candidate_padded_cell_polygon_verts, polygon_vertex_counts=candidate_cell_polygon_vertex_counts, line_coefficients=candidate_line_coefficients)
│   │   ├── calls _clip_convex_polygons_to_half_plane(polygon_verts=candidate_padded_cell_polygon_verts, polygon_vertex_counts=candidate_cell_polygon_vertex_counts, line_coefficients=-candidate_line_coefficients)
│   │   ├── calls _compute_convex_polygon_areas(polygon_verts=positive_polygon_verts, polygon_vertex_counts=positive_polygon_vertex_counts)
│   │   └── calls _compute_convex_polygon_areas(polygon_verts=negative_polygon_verts, polygon_vertex_counts=negative_polygon_vertex_counts)
│   └── calls _compute_convex_polygon_areas(polygon_verts=cell_polygon_verts, polygon_vertex_counts=cell_polygon_vertex_counts)
├── def _assign_visible_faces_to_cells(cell_polygon_verts: torch.Tensor, cell_polygon_vertex_counts: torch.Tensor, cell_pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_indices: torch.Tensor, pixel_face_valid_mask: torch.Tensor, pixel_depth_ordering_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Assign each batched arrangement cell to its frontmost covering face.
│   └── calls _compute_points_in_convex_polygons(points=cell_centroid.unsqueeze(1).expand(-1, candidate_polygon_verts.shape[1], -1).reshape(-1, 2), polygon_verts=candidate_polygon_verts.reshape(-1, candidate_polygon_verts.shape[2], 2), polygon_vertex_counts=candidate_polygon_vertex_counts.reshape(-1))
├── def _pack_face_pixel_polygons_by_pixel(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, clipped_pixel_indices: torch.Tensor, clipped_face_indices: torch.Tensor, face_depth_ordering_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Pack variable-count face-pixel polygons into pixel-major padded tensors.
│   ├── impls linear_pixel_indices = each clipped pair's (row, col) flattened into one index
│   ├── impls sorted_pair_indices = the pairs ordered by linear_pixel_indices
│   ├── impls pixel_group_counts = the run length of each distinct pixel in that order
│   ├── impls max_faces_per_pixel = the largest pixel_group_counts entry
│   ├── impls group_start_offsets = the exclusive prefix sum of pixel_group_counts
│   ├── impls pixel_indices = the (row, col) of each group's first pair
│   ├── impls group_indices = each sorted pair's own pixel group
│   ├── impls within_group_indices = each sorted pair's rank inside its own group
│   ├── impls pixel_polygon_verts = the sorted verts scattered into a zeroed [P, max_faces_per_pixel, V, 2] tensor at (group, rank)
│   ├── impls pixel_polygon_vertex_counts = the sorted counts scattered the same way
│   ├── impls pixel_face_indices = the sorted face indices scattered into a -1-filled tensor the same way
│   ├── impls pixel_face_valid_mask = pixel_face_indices at or above zero
│   ├── impls pixel_depth_ordering_coefficients = face_depth_ordering_coefficients of the sorted face indices, scattered the same way
│   └── return  # pixel_indices / pixel_polygon_verts / pixel_polygon_vertex_counts / pixel_face_indices / pixel_face_valid_mask / pixel_depth_ordering_coefficients, each contiguous
├── def compute_face_depth_ordering_coefficients(face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor
│   ├── # Compute the affine screen-space plane the occlusion test compares faces on: 1/z under a perspective model, z itself under an orthographic one, whose screen-to-surface map is affine and whose depth is therefore already linear in (x, y).
│   ├── def _validate_inputs [local]
│   │   └── if intrinsics.model in {"simple_pinhole", "pinhole"}
│   │       └── assert every face_vertex_depth is positive  # only these models go on to invert the depth, so the check is theirs rather than the function's
│   ├── calls _validate_inputs
│   ├── impls solve_matrix = face_screen_verts with a ones column appended  # the affine (x, y, 1) basis per corner
│   ├── assert every solve_matrix determinant is non-degenerate
│   ├── if intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   └── return the solve of solve_matrix against 1 / face_vertex_depth, contiguous  # [F, 3], the affine 1/z map's coefficients
│   ├── if intrinsics.model == "ortho"
│   │   └── return the solve of solve_matrix against -face_vertex_depth, contiguous  # [F, 3]; z is already affine in screen space here, negated so the frontmost face still scores highest
│   └── assert 0, "Should not reach here."
├── def camera_verts_to_pixel(verts_camera: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor
│   ├── # Project camera-space verts into image pixel coordinates through the intrinsics' own model, the one projection every consumer reaches rather than writing one model's formula out again.
│   ├── calls intrinsics.project(points_camera=verts_camera, inplace=False)  # [V, 2]; the perspective divide is the pinhole models' own, an ortho camera's project being a scale and an offset instead
│   └── return
├── def clip_convex_polygons_to_pixel_squares(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Clip convex polygons against their corresponding pixel squares.
│   ├── if torch.all(polygon_vertex_counts == 3)
│   │   └── calls _clip_triangle_polygons_to_pixel_squares(triangle_verts=polygon_verts[:, :3, :].contiguous(), pixel_x=pixel_x, pixel_y=pixel_y, output_vertex_capacity=polygon_verts.shape[1])
│   └── for coefficients in line_coefficients
│       └── calls _clip_convex_polygons_to_half_plane(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts, line_coefficients=coefficients)
├── def _clip_convex_polygons_to_half_plane(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, line_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Clip convex polygons against one half-plane.
│   ├── impls edge_active = the edge slots below each polygon's own vertex count
│   ├── impls next_indices = each edge's successor slot, wrapping to zero at that polygon's own count
│   ├── impls current_line_values = line_coefficients evaluated at current_verts
│   ├── impls next_line_values = line_coefficients evaluated at next_verts
│   ├── impls current_inside = the active edges whose current_line_values sit at or above zero
│   ├── impls next_inside = the active edges whose next_line_values sit at or above zero
│   ├── impls crossing_mask = the active edges whose two endpoints disagree on inside
│   ├── impls edge_t = current_line_values over current_line_values less next_line_values, on the non-degenerate crossings alone
│   ├── impls intersection_verts = current_verts advanced by edge_t toward next_verts
│   ├── impls candidate_verts = intersection_verts interleaved with next_verts
│   ├── impls candidate_vertex_valid_mask = crossing_mask interleaved with next_inside
│   ├── impls clipped_polygon_vertex_counts = the per-polygon count of valid candidates
│   ├── assert every clipped count fits the input vertex capacity
│   ├── impls clipped_polygon_verts = the valid candidates compacted into their own output slots, zero elsewhere
│   └── return clipped_polygon_verts, clipped_polygon_vertex_counts
├── def project_screen_polygons_to_face_uv(polygon_verts: torch.Tensor, face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, face_verts_uvs: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor
│   ├── # Map image-space polygon verts to UV: perspective-correct 1/z-weighted barycentrics under a perspective model, plain screen-space barycentrics under an orthographic one, an ortho projection carrying screen barycentrics onto surface barycentrics unchanged.
│   ├── calls _cross_2d(a=face_screen_v1 - face_screen_v0, b=face_screen_v2 - face_screen_v0)
│   ├── calls _cross_2d(a=face_screen_v1 - polygon_verts, b=face_screen_v2 - polygon_verts)
│   ├── calls _cross_2d(a=face_screen_v2 - polygon_verts, b=face_screen_v0 - polygon_verts)
│   ├── impls screen_barycentrics = the three signed sub-triangle areas each over the whole face's signed area
│   ├── if intrinsics.model in {"simple_pinhole", "pinhole"}
│   │   ├── impls surface_barycentrics = screen_barycentrics each over its own corner's face_vertex_depth, renormalized to sum to one
│   │   └── return the three face_verts_uvs corners weighted by surface_barycentrics, contiguous
│   ├── if intrinsics.model == "ortho"
│   │   └── return the three face_verts_uvs corners weighted by screen_barycentrics, contiguous  # the affine screen-to-surface map carries them across as the surface barycentrics themselves
│   └── assert 0, "Should not reach here."
├── def duplicate_wrapped_uv_polygons(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Duplicate UV polygons across the cylindrical wrap boundary when needed.
│   └── calls _compute_convex_polygon_bounds(polygon_verts=uv_polygon_verts, polygon_vertex_counts=uv_polygon_vertex_counts)
├── def build_uv_polygon_texel_intersections(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor, texture_size: int) -> torch.Tensor
│   ├── # Build exact UV-polygon to texel-cell intersection indices.
│   ├── calls _compute_convex_polygon_bounds(polygon_verts=polygon_texel_verts, polygon_vertex_counts=uv_polygon_vertex_counts)
│   └── while chunk_start < uv_polygon_verts.shape[0]
│       ├── calls _compute_points_in_convex_polygons(points=pixel_centers, polygon_verts=candidate_polygon_verts, polygon_vertex_counts=candidate_polygon_vertex_counts)
│       ├── calls _compute_points_near_convex_polygon_boundaries(points=pixel_centers, polygon_verts=candidate_polygon_verts, polygon_vertex_counts=candidate_polygon_vertex_counts, squared_distance_threshold=boundary_squared_distance_threshold)
│       └── if torch.any(boundary_candidate_mask)
│           └── if len(boundary_triangle_chunks) > 0
│               └── calls _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts=boundary_triangles, pixel_x=boundary_pixel_x[boundary_triangle_candidate_indices], pixel_y=boundary_pixel_y[boundary_triangle_candidate_indices])
├── def _compute_points_in_convex_polygons(points: torch.Tensor, polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   ├── # Test whether each point lies inside its corresponding convex polygon.
│   └── calls _cross_2d(a=next_verts - current_verts, b=points.reshape(-1, 1, 2) - current_verts)
├── def _compute_convex_polygon_bounds(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute axis-aligned bounds for convex polygons.
│   ├── if polygon_verts carries no polygon
│   │   └── return four empty tensors
│   ├── impls vertex_active_mask = the vertex slots below each polygon's own vertex count
│   ├── impls polygon_x_min = the per-polygon amin over x, the inactive slots held at +inf
│   ├── impls polygon_x_max = the per-polygon amax over x, the inactive slots held at -inf
│   ├── impls polygon_y_min = the per-polygon amin over y, the inactive slots held at +inf
│   ├── impls polygon_y_max = the per-polygon amax over y, the inactive slots held at -inf
│   └── return polygon_x_min, polygon_x_max, polygon_y_min, polygon_y_max, each contiguous
├── def _compute_points_near_convex_polygon_boundaries(points: torch.Tensor, polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, squared_distance_threshold: float) -> torch.Tensor
│   ├── # Test whether each point lies near its corresponding convex polygon boundary.
│   ├── if points carries no point
│   │   └── return an empty bool tensor
│   ├── impls edge_active_mask = the edge slots below each polygon's own vertex count
│   ├── impls next_indices = each edge's successor slot, wrapping to zero at that polygon's own count
│   ├── impls edge_vectors = next_verts less current_verts
│   ├── impls point_offsets = each point less its polygon's current_verts
│   ├── impls projection_t = point_offsets projected along edge_vectors over their squared length, zero where that length is degenerate, clamped into [0, 1]
│   ├── impls closest_points = current_verts plus projection_t along edge_vectors
│   ├── impls squared_distance = the squared distance from each point to closest_points, held at +inf on the inactive edges
│   └── return the per-point smallest squared_distance within squared_distance_threshold, contiguous  # one bool per point
├── def triangulate_convex_uv_polygons(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   ├── # Triangulate convex UV polygons into a triangle soup.
│   ├── impls uv_triangle_chunks = an empty list
│   ├── for each fan index from one to the vertex capacity less one
│   │   ├── impls fan_valid_mask = the polygons whose vertex count passes fan index + 1
│   │   ├── if no polygon is valid at this fan index
│   │   │   └── continue
│   │   └── impls append the valid polygons' (first, fan index, fan index + 1) corner triples to uv_triangle_chunks
│   ├── if uv_triangle_chunks is empty
│   │   └── return an empty (0, 3, 2) float32 tensor
│   ├── impls uv_triangles = uv_triangle_chunks concatenated
│   ├── impls uv_triangles = uv_triangles converted to float32
│   ├── impls uv_triangles = uv_triangles made contiguous
│   └── return uv_triangles
├── def build_uv_triangle_texel_intersections_v2(uv_triangles: torch.Tensor, texture_size: int) -> torch.Tensor
│   ├── # Build approximate step-2 `v2` UV-triangle to texel-cell intersections.
│   ├── def _compute_triangle_edge_function_coefficients(triangle_verts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Compute oriented triangle edge-function coefficients and thresholds.
│   │   ├── impls next_triangle_verts = each triangle's corners rotated by one
│   │   ├── impls edge_a = current y less next y
│   │   ├── impls edge_b = next x less current x
│   │   ├── impls edge_c = current x times next y less next x times current y
│   │   ├── impls triangle_orientation = plus one where each triangle's double area is non-negative, minus one elsewhere
│   │   ├── impls orient edge_a, edge_b and edge_c by triangle_orientation               # impls-node-one-step:skip  # so a point inside scores non-negative on every edge
│   │   ├── impls edge_thresholds = half the summed absolute oriented edge_a and edge_b  # impls-node-one-step:skip  # the half-texel margin each edge admits
│   │   ├── impls edge_function_coefficients = the oriented edge_a, edge_b, and edge_c stacked  # impls-node-one-step:skip
│   │   ├── impls edge_function_coefficients = edge_function_coefficients made contiguous
│   │   └── return edge_function_coefficients, edge_thresholds
│   ├── calls _compute_triangle_edge_function_coefficients(triangle_verts=triangle_texel_verts)
│   └── if torch.any(boundary_candidate_mask)
│       └── while boundary_chunk_start < boundary_candidate_indices.shape[0]
│           └── calls _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts=triangle_texel_verts[repeated_triangle_indices[boundary_chunk_indices]], pixel_x=pixel_x[boundary_chunk_indices], pixel_y=pixel_y[boundary_chunk_indices])
├── def _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor) -> torch.Tensor
│   ├── # Detect positive-area overlap between triangles and pixel squares.
│   ├── calls _clip_triangle_polygons_to_pixel_squares(triangle_verts=triangle_verts[bbox_overlap_mask], pixel_x=pixel_x[bbox_overlap_mask], pixel_y=pixel_y[bbox_overlap_mask], output_vertex_capacity=8)
│   └── calls _compute_convex_polygon_areas(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts)
├── def _compute_convex_polygon_areas(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   ├── # Compute areas of convex polygons.
│   ├── impls edge_active = the edge slots below each polygon's own vertex count
│   ├── impls next_indices = each edge's successor slot, wrapping to zero at that polygon's own count
│   ├── impls edge_term = current_verts' x times next_verts' y less current_verts' y times next_verts' x  # the shoelace term
│   ├── impls double_area = edge_term summed over the active edges alone
│   └── return half the absolute double_area, contiguous
├── def _clip_triangle_polygons_to_pixel_squares(triangle_verts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor, output_vertex_capacity: int) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Clip triangles against pixel squares with exact candidate-point geometry.
│   └── calls _compute_points_in_triangles(points=square_corners, triangle_verts=triangle_verts)
├── def _compute_points_in_triangles(points: torch.Tensor, triangle_verts: torch.Tensor) -> torch.Tensor
│   ├── # Test whether batched points lie inside their corresponding triangles.
│   ├── calls _cross_2d(a=(triangle_v1 - triangle_v0).expand(-1, point_count, -1), b=points - triangle_v0)
│   ├── calls _cross_2d(a=(triangle_v2 - triangle_v1).expand(-1, point_count, -1), b=points - triangle_v1)
│   └── calls _cross_2d(a=(triangle_v0 - triangle_v2).expand(-1, point_count, -1), b=points - triangle_v2)
└── def _cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
    ├── # Compute 2D cross product magnitude.
    └── return a's x times b's y less a's y times b's x, keeping the trailing axis  # the signed z of the 2D cross product
```

`models/three_d/meshes/texture/extract/visibility/texel_visibility_v2.py`

```text
texel_visibility_v2.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from models.three_d.meshes.texture.extract.camera_geometry import compute_points_in_front_of_camera
├── from models.three_d.meshes.texture.extract.weights.normal_weights import compute_f_normals_weights
├── from models.three_d.point_cloud.ops.world_to_camera_transform import world_to_camera_transform
├── FRONT_DEPTH_GAP_LOG_MAD_MULTIPLIER
├── def compute_f_visibility_mask_v2(verts: torch.Tensor, faces: torch.Tensor, face_verts_uvs: torch.Tensor, camera: Cameras, image_height: int, image_width: int, texel_face_map: Dict[str, torch.Tensor]) -> torch.Tensor
│   ├── # Compute one-view UV-pixel visibility mask from projected texel centers.
│   ├── calls _map_valid_texels_to_continuous_uv_coords(valid_texel_mask=valid_texel_mask)
│   ├── calls _map_continuous_uv_coords_to_barycentric_coords(continuous_uv_coords=continuous_uv_coords, valid_texel_indices=valid_texel_indices, face_verts_uvs=face_verts_uvs, texel_face_map=texel_face_map)
│   ├── calls _filter_texels_by_face_facing(valid_texel_indices=valid_texel_indices, texel_face_indices=texel_face_indices, barycentric_coords=barycentric_coords, verts=verts, faces=faces, camera=camera)
│   ├── calls _map_barycentric_coords_to_3d_world_coords(barycentric_coords=barycentric_coords, texel_face_indices=texel_face_indices, verts=verts, faces=faces)
│   ├── calls _compute_mesh_diagonal(verts=verts)
│   └── calls _compute_texel_visibility_mask_from_world_coords(world_coords=world_coords, valid_texel_indices=valid_texel_indices, valid_texel_mask=valid_texel_mask, mesh_diagonal=mesh_diagonal, camera=camera, image_height=image_height, image_width=image_width)
├── def _map_valid_texels_to_continuous_uv_coords(valid_texel_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Map valid texel centers to continuous UV coordinates.
│   ├── impls valid_texel_indices = the (row, col) positions where valid_texel_mask passes a half
│   ├── assert valid_texel_indices is rank 2 with two columns
│   ├── if no texel is valid
│   │   ├── impls valid_texel_indices = an empty index tensor
│   │   ├── impls continuous_uv_coords = an empty coordinate tensor
│   │   └── return valid_texel_indices, continuous_uv_coords
│   ├── impls continuous_u = the texel's column plus a half, over the texture width
│   ├── impls continuous_v = the texel's row plus a half, over the texture height
│   ├── impls continuous_uv_coords = continuous_u stacked against continuous_v, contiguous
│   ├── impls valid_texel_indices = valid_texel_indices contiguous
│   └── return valid_texel_indices, continuous_uv_coords
├── def _map_continuous_uv_coords_to_barycentric_coords(continuous_uv_coords: torch.Tensor, valid_texel_indices: torch.Tensor, face_verts_uvs: torch.Tensor, texel_face_map: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Map continuous UV coordinates to owning-face barycentric coordinates.
│   ├── calls _wrap_continuous_uv_coords_for_faces(continuous_uv_coords=continuous_uv_coords, face_verts_uvs=face_verts_uvs)
│   └── calls _compute_barycentric_coords_in_uv_faces(continuous_uv_coords=wrapped_continuous_uv_coords, face_verts_uvs=face_verts_uvs)
├── def _filter_texels_by_face_facing(valid_texel_indices: torch.Tensor, texel_face_indices: torch.Tensor, barycentric_coords: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, camera: Cameras) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Filter texels whose owning mesh face is back-facing in the current view.
│   └── calls compute_f_normals_weights(mesh=Mesh(verts=verts, faces=faces), camera=camera, weights_cfg={'weights': 'normals'})
├── def _map_barycentric_coords_to_3d_world_coords(barycentric_coords: torch.Tensor, texel_face_indices: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor
│   ├── # Map barycentric texel coordinates to world-space mesh points.
│   ├── if barycentric_coords is empty
│   │   └── return an empty (0, 3) float32 tensor
│   ├── impls face_verts = verts at faces of texel_face_indices
│   ├── impls world_coords = the three face_verts corners each weighted by its own barycentric column
│   └── return world_coords, contiguous  # [N, 3]
├── def _compute_texel_visibility_mask_from_world_coords(world_coords: torch.Tensor, valid_texel_indices: torch.Tensor, valid_texel_mask: torch.Tensor, mesh_diagonal: float, camera: Cameras, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Compute texel visibility by keeping the front depth-prefix per pixel.
│   ├── calls world_to_camera_transform(points=world_coords, extrinsics=camera_single.extrinsics.extrinsics, inplace=False)
│   ├── calls camera_single.intrinsics.project
│   ├── calls compute_points_in_front_of_camera(points_camera=texel_camera_coords, intrinsics=camera_single.intrinsics)
│   ├── impls projection_valid_mask = the in-front texels whose projected pixel also lands inside the image
│   └── calls _select_visible_depth_clusters_per_camera_pixel(linear_pixel_indices=visible_linear_pixel_indices, depth=visible_projected_depth, mesh_diagonal=mesh_diagonal)
├── def _wrap_continuous_uv_coords_for_faces(continuous_uv_coords: torch.Tensor, face_verts_uvs: torch.Tensor) -> torch.Tensor
│   ├── # Wrap texel-center UV coordinates into the seam-safe face-local chart.
│   ├── if continuous_uv_coords is empty
│   │   └── return continuous_uv_coords
│   ├── impls seam_face_mask = the faces whose largest UV u passes one  # the seam-safe chart carries them beyond the unit square
│   ├── if any face is a seam face
│   │   └── impls add one wrap to those texels' u wherever it sits below a half
│   └── return the wrapped coordinates, contiguous
├── def _compute_barycentric_coords_in_uv_faces(continuous_uv_coords: torch.Tensor, face_verts_uvs: torch.Tensor) -> torch.Tensor
│   ├── # Compute barycentric coordinates of points inside UV triangles.
│   ├── if continuous_uv_coords is empty
│   │   └── return an empty (0, 3) float32 tensor
│   ├── impls edge01 = the face's second UV corner less its first
│   ├── impls edge02 = the face's third UV corner less its first
│   ├── impls point_offset = continuous_uv_coords less the face's first UV corner
│   ├── impls determinant = the 2D cross product of edge01 with edge02
│   ├── assert every determinant is non-degenerate  # every owning UV triangle carries area
│   ├── impls barycentric1 = the cross product of point_offset with edge02, over determinant
│   ├── impls barycentric2 = the cross product of edge01 with point_offset, over determinant
│   ├── impls barycentric0 = one less barycentric1 less barycentric2
│   └── return the three stacked into columns, contiguous  # [N, 3]
├── def _compute_mesh_diagonal(verts: torch.Tensor) -> float
│   ├── # Compute the full-mesh diagonal length.
│   ├── impls min_verts = the per-axis minimum over verts
│   ├── impls max_verts = the per-axis maximum over verts
│   ├── impls mesh_diagonal = the norm of max_verts less min_verts, as a float
│   ├── assert mesh_diagonal is positive
│   └── return mesh_diagonal
├── def _select_visible_depth_clusters_per_camera_pixel(linear_pixel_indices: torch.Tensor, depth: torch.Tensor, mesh_diagonal: float) -> torch.Tensor
│   ├── # Keep only the first front depth cluster in each pixel stack.
│   ├── calls _sort_depth_stacks_per_camera_pixel(linear_pixel_indices=linear_pixel_indices, depth=depth)
│   └── calls _compute_front_depth_gap_threshold_relative(sorted_depth=sorted_depth, segment_start_mask=segment_start_mask, mesh_diagonal=mesh_diagonal)
├── def _sort_depth_stacks_per_camera_pixel(linear_pixel_indices: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Sort projected texels into per-pixel depth stacks.
│   ├── if linear_pixel_indices is empty
│   │   └── return four empty tensors
│   ├── impls depth_order = the entries stably ordered by depth
│   ├── impls pixel_order = the stable ordering of linear_pixel_indices taken in depth_order  # the stable second pass keeps each pixel's stack depth-ascending
│   ├── impls sorted_indices = depth_order taken through pixel_order
│   ├── impls sorted_linear_pixel_indices = linear_pixel_indices at sorted_indices
│   ├── impls sorted_depth = depth at sorted_indices
│   ├── impls segment_start_mask = the entries whose pixel differs from their predecessor's
│   └── return sorted_indices, sorted_linear_pixel_indices, sorted_depth, segment_start_mask, each contiguous
└── def _compute_front_depth_gap_threshold_relative(sorted_depth: torch.Tensor, segment_start_mask: torch.Tensor, mesh_diagonal: float) -> float
    ├── # Derive the front-depth stopping threshold from the gap distribution.
    ├── if sorted_depth carries at most one entry
    │   └── return 0.0
    ├── impls relative_depth_gap_from_previous = each entry's depth step from its predecessor, over mesh_diagonal
    ├── impls positive_relative_depth_gaps = the strictly positive gaps inside a pixel's own stack  # a segment start is a different pixel, not a gap
    ├── if positive_relative_depth_gaps is empty
    │   └── return 0.0
    ├── impls log_positive_relative_depth_gaps = the base-10 logarithm of positive_relative_depth_gaps
    ├── impls log_gap_median = the median of log_positive_relative_depth_gaps
    ├── impls log_gap_mad = the median absolute deviation about log_gap_median
    ├── impls log_gap_threshold = log_gap_median plus FRONT_DEPTH_GAP_LOG_MAD_MULTIPLIER times log_gap_mad
    ├── impls front_depth_gap_threshold = ten raised to log_gap_threshold, as a Python float
    └── return front_depth_gap_threshold  # a wider relative gap closes the front cluster
```

`models/three_d/meshes/texture/extract/visibility/vertex_visibility.py`

```text
vertex_visibility.py
├── import torch
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
├── from data.structures.three_d.mesh.mesh import Mesh
├── from models.three_d.meshes.texture.extract.camera_geometry import compute_camera_view_directions, compute_points_in_front_of_camera, project_verts_to_image, render_camera_face_index_buffer
├── def compute_v_visibility_mask(mesh: Mesh, camera: Cameras, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Compute one-view binary visibility mask over verts.
│   ├── calls project_verts_to_image(verts=mesh.verts, camera=camera, image_height=image_height, image_width=image_width)
│   └── calls _compute_rasterized_visible_vertex_mask(verts_camera=verts_camera, faces=mesh.faces.to(device=mesh.device, dtype=torch.long).contiguous(), intrinsics=camera[0].intrinsics, image_height=image_height, image_width=image_width)
├── def _compute_rasterized_visible_vertex_mask(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: CameraIntrinsics, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Compute rasterized one-view vertex visibility mask.
│   ├── calls compute_points_in_front_of_camera(points_camera=verts_camera, intrinsics=intrinsics)
│   ├── if no vertex sits in front of the camera
│   │   ├── impls visible_vertex_mask = an all-false mask over the verts
│   │   └── return visible_vertex_mask
│   ├── calls _compute_face_front_facing_mask(verts_camera=verts_camera, faces=faces, intrinsics=intrinsics)
│   ├── if no face is front-facing
│   │   ├── impls visible_vertex_mask = an all-false mask over the verts
│   │   └── return visible_vertex_mask
│   ├── calls render_camera_face_index_buffer(verts_camera=verts_camera, faces=front_facing_faces, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
│   ├── impls visible_vertex_mask = the verts the face-index buffer reached
│   ├── impls visible_vertex_mask = visible_vertex_mask intersected with the in-front mask
│   └── return visible_vertex_mask
└── def _compute_face_front_facing_mask(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor
    ├── # Compute which camera-space mesh faces are front-facing.
    ├── impls face_normals_camera = the cross product of each face's two edge vectors
    ├── assert every face normal carries a non-zero magnitude
    ├── impls face_normals_camera = face_normals_camera over its own row norms
    ├── impls face_centers_camera = the mean of each face's three camera-space corners
    ├── calls compute_camera_view_directions(points_camera=face_centers_camera, intrinsics=intrinsics)
    ├── impls alignment = the dot product of face_normals_camera with face_view_direction
    └── return the faces whose alignment is strictly positive
```
