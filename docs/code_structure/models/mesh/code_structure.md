# Mesh Texture Extraction Code Structure

Code-structure skeleton for `models/three_d/meshes/texture/extract/`. `def` lines carry full prototypes with declaration-level type annotations; `calls` lines carry keyword-form call arguments. `_validate_inputs` / `_normalize_inputs` local helpers are not skeleton nodes — they are a uniform implementation-time pattern — so neither they nor the `calls` confined to them appear. Every cross-file `calls` target is a bare symbol resolving to a named import at the top of its file's block.

## 1. Code structure trees

```text
models/three_d/meshes/texture/extract/camera_geometry.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.point_cloud.camera.transform import world_to_camera_transform
├── def _render_camera_face_index_buffer(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Render a one-view camera-space face-index buffer.
│   └── calls _camera_verts_to_clip(verts_camera=verts_camera, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
├── def _render_camera_depth_buffer(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Render a one-view camera-space depth buffer.
│   └── calls _camera_verts_to_clip(verts_camera=verts_camera, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
├── def _verts_world_to_camera(verts: torch.Tensor, camera: Cameras) -> torch.Tensor
│   ├── # Transform one-view world-space verts to camera-space verts.
│   └── calls world_to_camera_transform(points=verts, extrinsics=camera_single.extrinsics, inplace=False)
├── def _project_verts_to_image(verts: torch.Tensor, camera: Cameras, image_height: int, image_width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Project world-space verts to image pixels for one view.
│   └── calls _verts_world_to_camera(verts=verts, camera=camera)
└── def _camera_verts_to_clip(verts_camera: torch.Tensor, intrinsics: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor
    └── # Convert camera-space verts to clip-space for rasterization.
```

```text
models/three_d/meshes/texture/extract/extract.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.validate_vertex_color import validate_vertex_color
├── from models.three_d.meshes.texture.extract.camera_geometry import _project_verts_to_image
├── from models.three_d.meshes.texture.extract.normal_weights import _compute_f_normals_weights, _compute_v_normals_weights
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility import compute_f_visibility_mask
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import compute_f_visibility_mask_v2
├── from models.three_d.meshes.texture.extract.visibility.vertex_visibility import compute_v_visibility_mask
├── def _validate_rgb_image(obj: Any) -> None
│   └── # Validate that an object is an RGB image tensor (CHW/HWC/NCHW/NHWC, uint8 [0,255] or float32 [0,1]).
├── def extract_texture_from_images(mesh: Union[Mesh, List[Mesh]], images: Union[torch.Tensor, List[torch.Tensor]], cameras: Cameras, weights_cfg: Dict[str, Any]={}, texture_size: int=1024, default_color: float=0.7, return_valid_mask: bool=False, texel_visibility_method: str='v1', polygon_rast_method: str='v2') -> Union[torch.Tensor, Dict[str, torch.Tensor]]
│   ├── # Extract texture from multi-view RGB images.
│   ├── if not extract_uv_texture_map
│   │   └── calls _extract_vertex_color_from_images(meshes=meshes, images_nchw=images_nchw, cameras=cameras, weights_cfg=weights_cfg, default_color=default_color)
│   └── calls _extract_uv_texture_map_from_images(meshes=meshes, images_nchw=images_nchw, cameras=cameras, weights_cfg=weights_cfg, texture_size=texture_size, default_color=default_color, texel_visibility_method=texel_visibility_method, polygon_rast_method=polygon_rast_method)
├── def _extract_vertex_color_from_images(meshes: List[Mesh], images_nchw: torch.Tensor, cameras: Cameras, weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Fuse per-view projected vertex colors into one vertex-color tensor.
│   ├── for view_idx in range(images_nchw.shape[0])
│   │   └── calls _extract_vertex_color_from_single_image(mesh=meshes[view_idx], image=images_nchw[view_idx], camera=cameras[view_idx:view_idx + 1], weights_cfg=weights_cfg, default_color=default_color)
│   └── calls _fuse_vertex_color_observations(observations=observations, weights_cfg=weights_cfg, default_color=default_color)
├── def _fuse_vertex_color_observations(observations: List[Dict[str, torch.Tensor]], weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Fuse one-view vertex-color observations into one vertex-color tensor.
│   ├── if multi_view_robustness == 'none'
│   ├── else
│   │   └── calls validate_vertex_color(obj=provisional_vertex_color)
│   ├── calls validate_vertex_color(obj=vertex_color)
│   └── calls validate_vertex_color(obj=vertex_color)
├── def _extract_vertex_color_from_single_image(mesh: Mesh, image: torch.Tensor, camera: Cameras, weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Extract one-view vertex colors and corresponding per-vertex weights.
│   ├── calls compute_v_visibility_mask(mesh=mesh, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]))
│   ├── if weights == 'normals'
│   │   └── calls _compute_v_normals_weights(mesh=mesh, camera=camera, weights_cfg=weights_cfg)
│   ├── else
│   └── calls _project_v_colors(mesh=mesh, image=image, camera=camera, default_color=default_color)
├── def _project_v_colors(mesh: Mesh, image: torch.Tensor, camera: Cameras, default_color: float) -> torch.Tensor
│   ├── # Project one image to verts and sample per-vertex RGB colors.
│   └── calls _project_verts_to_image(verts=mesh.verts, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]))
├── def _extract_uv_texture_map_from_images(meshes: List[Mesh], images_nchw: torch.Tensor, cameras: Cameras, weights_cfg: Dict[str, Any], texture_size: int, default_color: float, texel_visibility_method: str, polygon_rast_method: str='v2') -> Dict[str, torch.Tensor]
│   ├── # Fuse per-view UV observations into one UV texture map.
│   ├── calls _build_uv_rasterization_data(mesh=reference_mesh, texture_size=texture_size)
│   ├── for view_idx in range(images_nchw.shape[0])
│   │   └── calls _extract_uv_texture_map_from_single_image(mesh=meshes[view_idx], image=images_nchw[view_idx], camera=cameras[view_idx:view_idx + 1], weights_cfg=weights_cfg, uv_rasterization_data=uv_rasterization_data, texel_visibility_method=texel_visibility_method, polygon_rast_method=polygon_rast_method)
│   └── calls _fuse_uv_texture_observations(observations=observations, weights_cfg=weights_cfg, default_color=default_color)
├── def _fuse_uv_texture_observations(observations: List[Dict[str, torch.Tensor]], weights_cfg: Dict[str, Any], default_color: float) -> Dict[str, torch.Tensor]
│   ├── # Fuse one-view UV observations into one UV texture map.
│   ├── if multi_view_robustness == 'none'
│   ├── else
│   │   └── calls _validate_rgb_image(obj=provisional_uv_texture_map)
│   ├── calls _validate_rgb_image(obj=uv_texture_map)
│   └── calls _validate_rgb_image(obj=uv_texture_map)
├── def _extract_uv_texture_map_from_single_image(mesh: Mesh, image: torch.Tensor, camera: Cameras, weights_cfg: Dict[str, Any], uv_rasterization_data: Dict[str, torch.Tensor], texel_visibility_method: str='v1', polygon_rast_method: str='v2') -> Dict[str, torch.Tensor]
│   ├── # Extract one-view UV texture observation and UV weight map.
│   ├── if texel_visibility_method == 'v1'
│   │   └── calls compute_f_visibility_mask(verts=mesh.verts, faces=mesh.faces, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]), uv_rasterization_data=uv_rasterization_data, polygon_rast_method=polygon_rast_method)
│   ├── else
│   │   └── calls compute_f_visibility_mask_v2(verts=mesh.verts, faces=mesh.faces, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]), uv_rasterization_data=uv_rasterization_data)
│   ├── if weights == 'normals'
│   │   ├── calls _compute_f_normals_weights(mesh=mesh, camera=camera, weights_cfg=weights_cfg)
│   │   └── calls _rasterize_face_weights_to_uv(face_weight=face_normals_weight, uv_rasterization_data=uv_rasterization_data)
│   ├── else
│   └── calls _project_f_colors(mesh=mesh, image=image, camera=camera, uv_rasterization_data=uv_rasterization_data)
├── def _project_f_colors(mesh: Mesh, image: torch.Tensor, camera: Cameras, uv_rasterization_data: Dict[str, torch.Tensor]) -> torch.Tensor
│   ├── # Project one image into UV space using rasterized UV correspondence.
│   ├── def _interpolate_uv_texel_image_coords(projected_vertex_xy: torch.Tensor, uv_rasterization_data: Dict[str, torch.Tensor]) -> torch.Tensor [local]
│   │   └── # Interpolate image-space coordinates for every occupied UV texel.
│   ├── def _sample_uv_texel_colors_from_source_image(interpolated_uv_xy: torch.Tensor, image: torch.Tensor) -> torch.Tensor [local]
│   │   └── # Sample source-image colors at interpolated UV texel image coordinates.
│   ├── calls _project_verts_to_image(verts=mesh.verts, camera=camera, image_height=int(image.shape[1]), image_width=int(image.shape[2]))
│   ├── calls _interpolate_uv_texel_image_coords(projected_vertex_xy=xy, uv_rasterization_data=uv_rasterization_data)
│   ├── calls _sample_uv_texel_colors_from_source_image(interpolated_uv_xy=interpolated_uv_xy, image=image)
│   └── calls _validate_rgb_image(obj=uv_texture)
├── def _build_uv_rasterization_data(mesh: Mesh, texture_size: int) -> Dict[str, torch.Tensor]
│   ├── # Build reusable UV rasterization tensors for UV-space operations.
│   ├── calls _build_uv_rasterization_mesh(verts_uvs=verts_uvs, faces=faces, faces_uvs=faces_uvs)
│   ├── calls _build_camera_uv_interpolation_data(verts_uvs=verts_uvs, faces=faces, faces_uvs=faces_uvs)
│   └── calls _verts_uvs_to_clip(verts_uvs=raster_verts_uvs)
├── def _build_uv_rasterization_mesh(verts_uvs: torch.Tensor, faces: torch.Tensor, faces_uvs: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]
│   ├── # Build a seam-safe UV triangle soup for UV rasterization.
│   ├── def _append_triangles(face_indices: torch.Tensor, face_uv: torch.Tensor) -> None [local]
│   │   └── # Append one batch of UV triangles to the triangle-soup buffers.
│   ├── calls _append_triangles(face_indices=non_seam_face_indices, face_uv=face_verts_uvs[non_seam_face_indices])
│   └── if seam_face_indices.numel() > 0
│       ├── calls _append_triangles(face_indices=seam_face_indices, face_uv=seam_face_uv)
│       └── calls _append_triangles(face_indices=seam_face_indices, face_uv=seam_face_uv_wrapped)
├── def _build_camera_uv_interpolation_data(verts_uvs: torch.Tensor, faces: torch.Tensor, faces_uvs: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]
│   └── # Build seam-safe per-face UV attributes for camera-space interpolation.
├── def _verts_uvs_to_clip(verts_uvs: torch.Tensor) -> torch.Tensor
│   └── # Convert UV coordinates to clip-space positions for UV rasterization.
└── def _rasterize_face_weights_to_uv(face_weight: torch.Tensor, uv_rasterization_data: Dict[str, torch.Tensor]) -> torch.Tensor
    └── # Map per-face weights to per-UV-pixel weights for one view.
```

```text
models/three_d/meshes/texture/extract/normal_weights.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.mesh.mesh import Mesh
├── from models.three_d.meshes.ops.normals import compute_vertex_normals
├── from models.three_d.meshes.texture.extract.camera_geometry import _verts_world_to_camera
├── def _compute_v_normals_weights(mesh: Mesh, camera: Cameras, weights_cfg: Dict[str, Any]) -> torch.Tensor
│   ├── # Compute one-view per-vertex normal-alignment weights.
│   ├── calls _verts_world_to_camera(verts=mesh.verts, camera=camera)
│   └── calls compute_vertex_normals(base_verts=verts_camera, faces=mesh.faces)
└── def _compute_f_normals_weights(mesh: Mesh, camera: Cameras, weights_cfg: Dict[str, Any]) -> torch.Tensor
    ├── # Compute one-view per-face normal-alignment weights.
    └── calls _verts_world_to_camera(verts=mesh.verts, camera=camera)
```

```text
models/three_d/meshes/texture/extract/weights_cfg.py
├── WEIGHTS_CFG_ALLOWED_KEYS
├── def validate_weights_cfg(weights_cfg: Dict[str, Any]) -> None
│   └── # Validate one texture-extraction weights config.
└── def normalize_weights_cfg(weights_cfg: Dict[str, Any], default_weights: str) -> Dict[str, Any]
    └── # Normalize one texture-extraction weights config.
```

```text
models/three_d/meshes/texture/extract/visibility/texel_visibility.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from models.three_d.meshes.texture.extract.camera_geometry import _verts_world_to_camera
├── from models.three_d.meshes.texture.extract.normal_weights import _compute_f_normals_weights
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility_geometry import _build_uv_polygon_texel_intersections, _build_uv_triangle_texel_intersections_v2, _build_visible_face_pixel_polygons, _camera_verts_to_pixel, _clip_convex_polygons_to_pixel_squares, _compute_convex_polygon_areas, _compute_face_inverse_depth_coefficients, _duplicate_wrapped_uv_polygons, _project_screen_polygons_to_face_uv, _triangulate_convex_uv_polygons
├── def compute_f_visibility_mask(verts: torch.Tensor, faces: torch.Tensor, camera: Cameras, image_height: int, image_width: int, uv_rasterization_data: Dict[str, torch.Tensor], polygon_rast_method: str='v2') -> torch.Tensor
│   ├── # Compute one-view UV-pixel visibility mask from exact camera-pixel footprints.
│   ├── calls _verts_world_to_camera(verts=verts, camera=camera)
│   ├── calls _compute_f_normals_weights(mesh=Mesh(verts=verts, faces=faces), camera=camera, weights_cfg={'weights': 'normals'})
│   ├── calls _compute_visible_uv_polygon_regions_from_camera_pixels(verts_camera=verts_camera, faces=faces, intrinsics=camera[0].intrinsics, image_height=image_height, image_width=image_width, face_front_facing_mask=face_front_facing_mask, camera_face_verts_uvs=camera_attr_verts_uvs.reshape(-1, 3, 2))
│   └── calls _compute_visible_uv_texels_from_uv_polygon_regions(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts, texture_size=int(uv_mask.shape[1]), polygon_rast_method=polygon_rast_method)
├── def _compute_visible_uv_polygon_regions_from_camera_pixels(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: torch.Tensor, image_height: int, image_width: int, face_front_facing_mask: torch.Tensor, camera_face_verts_uvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Compute exact visible UV polygon regions from camera pixels.
│   ├── calls _camera_verts_to_pixel(verts_camera=verts_camera, intrinsics=intrinsics)
│   ├── calls _compute_visible_screen_space_polygon_regions_inside_camera_pixels(face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, image_height=image_height, image_width=image_width)
│   └── calls _map_visible_screen_space_polygon_regions_to_uv(visible_screen_polygon_verts=visible_screen_polygon_verts, visible_screen_polygon_vertex_counts=visible_screen_polygon_vertex_counts, visible_screen_polygon_face_indices=visible_screen_polygon_face_indices, face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, face_verts_uvs=face_verts_uvs)
├── def _compute_visible_screen_space_polygon_regions_inside_camera_pixels(face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute exact visible screen-space polygon regions inside each camera pixel.
│   ├── calls _compute_face_pixel_polygon_intersections_without_occlusion(face_screen_verts=face_screen_verts, image_height=image_height, image_width=image_width)
│   └── calls _compute_visible_screen_space_polygon_regions_with_occlusion(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, clipped_pixel_indices=clipped_pixel_indices, clipped_face_indices=clipped_face_indices, face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth, image_height=image_height, image_width=image_width)
├── def _compute_face_pixel_polygon_intersections_without_occlusion(face_screen_verts: torch.Tensor, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute all face-pixel polygon intersections without considering occlusion.
│   ├── def _compute_projected_face_pixel_bounds() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] [local]
│   │   └── # Compute candidate pixel bounds for each projected face.
│   ├── def _enumerate_candidate_face_pixel_pairs(pair_count_per_face: torch.Tensor, pixel_x_start: torch.Tensor, pixel_y_start: torch.Tensor, pixel_x_count: torch.Tensor, total_pair_count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   └── # Enumerate all candidate face-pixel pairs.
│   ├── def _clip_face_triangles_to_pixel_squares(repeated_face_indices: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor, total_pair_count: int) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Clip projected face triangles to candidate pixel squares.
│   │   └── calls _clip_convex_polygons_to_pixel_squares(polygon_verts=polygon_verts, polygon_vertex_counts=polygon_vertex_counts, pixel_x=pixel_x.to(dtype=torch.float32), pixel_y=pixel_y.to(dtype=torch.float32))
│   ├── def _pack_valid_face_pixel_polygons(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor, repeated_face_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   ├── # Reject degenerate overlaps and pack the surviving polygons.
│   │   └── calls _compute_convex_polygon_areas(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts)
│   ├── calls _compute_projected_face_pixel_bounds()
│   ├── calls _enumerate_candidate_face_pixel_pairs(pair_count_per_face=pair_count_per_face, pixel_x_start=pixel_x_start, pixel_y_start=pixel_y_start, pixel_x_count=pixel_x_count, total_pair_count=total_pair_count)
│   ├── calls _clip_face_triangles_to_pixel_squares(repeated_face_indices=repeated_face_indices, pixel_x=pixel_x, pixel_y=pixel_y, total_pair_count=total_pair_count)
│   └── calls _pack_valid_face_pixel_polygons(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, pixel_x=pixel_x, pixel_y=pixel_y, repeated_face_indices=repeated_face_indices)
├── def _compute_visible_screen_space_polygon_regions_with_occlusion(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, clipped_pixel_indices: torch.Tensor, clipped_face_indices: torch.Tensor, face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Compute inter-polygon occlusion and remove hidden screen-space regions.
│   ├── def _compute_projected_face_inverse_depth_coefficients() -> torch.Tensor [local]
│   │   ├── # Compute affine inverse-depth coefficients for the projected faces.
│   │   └── calls _compute_face_inverse_depth_coefficients(face_screen_verts=face_screen_verts, face_vertex_depth=face_vertex_depth)
│   ├── def _build_exact_visible_face_pixel_polygons(face_inverse_depth_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   ├── # Resolve exact visible face-pixel polygons from clipped overlaps.
│   │   └── calls _build_visible_face_pixel_polygons(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, clipped_pixel_indices=clipped_pixel_indices, clipped_face_indices=clipped_face_indices, face_inverse_depth_coefficients=face_inverse_depth_coefficients)
│   ├── def _pack_visible_polygon_outputs(visible_polygon_verts: torch.Tensor, visible_polygon_vertex_counts: torch.Tensor, visible_polygon_face_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   └── # Pack exact visible polygons into the downstream tensor format.
│   ├── calls _compute_projected_face_inverse_depth_coefficients()
│   ├── calls _build_exact_visible_face_pixel_polygons(face_inverse_depth_coefficients=face_inverse_depth_coefficients)
│   └── calls _pack_visible_polygon_outputs(visible_polygon_verts=visible_polygon_verts, visible_polygon_vertex_counts=visible_polygon_vertex_counts, visible_polygon_face_indices=visible_polygon_face_indices)
├── def _map_visible_screen_space_polygon_regions_to_uv(visible_screen_polygon_verts: torch.Tensor, visible_screen_polygon_vertex_counts: torch.Tensor, visible_screen_polygon_face_indices: torch.Tensor, face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, face_verts_uvs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Map visible screen-space polygon regions into UV.
│   ├── def _gather_visible_polygon_face_geometry() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] [local]
│   │   └── # Gather owning-face geometry for each visible polygon.
│   ├── def _project_screen_polygon_verts_to_uv(polygon_face_screen_verts: torch.Tensor, polygon_face_vertex_depth: torch.Tensor, polygon_face_verts_uvs: torch.Tensor) -> torch.Tensor [local]
│   │   ├── # Project visible screen polygons into UV.
│   │   └── calls _project_screen_polygons_to_face_uv(polygon_verts=visible_screen_polygon_verts, face_screen_verts=polygon_face_screen_verts, face_vertex_depth=polygon_face_vertex_depth, face_verts_uvs=polygon_face_verts_uvs)
│   ├── def _pack_visible_uv_polygons(uv_polygon_verts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] [local]
│   │   └── # Pack UV polygons with their original vertex counts.
│   ├── calls _gather_visible_polygon_face_geometry()
│   ├── calls _project_screen_polygon_verts_to_uv(polygon_face_screen_verts=polygon_face_screen_verts, polygon_face_vertex_depth=polygon_face_vertex_depth, polygon_face_verts_uvs=polygon_face_verts_uvs)
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
│   │   └── calls _duplicate_wrapped_uv_polygons(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts)
│   ├── calls _duplicate_wrap_crossing_polygons()
│   └── calls _build_uv_polygon_texel_intersections(uv_polygon_verts=wrapped_uv_polygon_verts, uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts, texture_size=texture_size)
└── def _compute_uv_polygon_texel_contributions_v2(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor, texture_size: int) -> torch.Tensor
    ├── # Construct approximate step-2 `v2` texel contributions for visible UV polygons.
    ├── def _duplicate_wrap_crossing_polygons() -> Tuple[torch.Tensor, torch.Tensor] [local]
    │   ├── # Duplicate wrap-crossing polygons so the cylindrical UV union is preserved.
    │   └── calls _duplicate_wrapped_uv_polygons(uv_polygon_verts=uv_polygon_verts, uv_polygon_vertex_counts=uv_polygon_vertex_counts)
    ├── def _triangulate_wrapped_uv_polygons(wrapped_uv_polygon_verts: torch.Tensor, wrapped_uv_polygon_vertex_counts: torch.Tensor) -> torch.Tensor [local]
    │   ├── # Triangulate wrapped convex UV polygons into a triangle soup.
    │   └── calls _triangulate_convex_uv_polygons(polygon_verts=wrapped_uv_polygon_verts, polygon_vertex_counts=wrapped_uv_polygon_vertex_counts)
    ├── calls _duplicate_wrap_crossing_polygons()
    ├── calls _triangulate_wrapped_uv_polygons(wrapped_uv_polygon_verts=wrapped_uv_polygon_verts, wrapped_uv_polygon_vertex_counts=wrapped_uv_polygon_vertex_counts)
    └── calls _build_uv_triangle_texel_intersections_v2(uv_triangles=wrapped_uv_triangles, texture_size=texture_size)
```

```text
models/three_d/meshes/texture/extract/visibility/texel_visibility_geometry.py
├── TARGET_MULTI_FACE_PIXEL_SPLIT_LINE_BUDGET
├── def _plan_multi_face_pixel_chunks(face_count_per_pixel: torch.Tensor, max_verts_per_polygon: int, target_split_line_budget: int) -> List[Tuple[int, int]]
│   └── # Plan sorted multi-face pixel chunks under a split-line budget.
├── def _gather_visible_pixel_face_polygons(pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_indices: torch.Tensor, pixel_face_slot_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   └── # Gather selected pixel-face polygons into flat visible outputs.
├── def _compute_pair_positive_area_overlap_mask(first_polygon_verts: torch.Tensor, first_polygon_vertex_counts: torch.Tensor, second_polygon_verts: torch.Tensor, second_polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   └── # Detect positive-area overlap for convex polygon pairs.
├── def _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor) -> torch.Tensor
│   ├── # Detect positive-area overlap between triangles and pixel squares.
│   ├── calls _clip_triangle_polygons_to_pixel_squares(triangle_verts=triangle_verts[bbox_overlap_mask], pixel_x=pixel_x[bbox_overlap_mask], pixel_y=pixel_y[bbox_overlap_mask], output_vertex_capacity=8)
│   └── calls _compute_convex_polygon_areas(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts)
├── def _compute_multi_face_pixel_second_bucket_mask(pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_valid_mask: torch.Tensor) -> torch.Tensor
│   ├── # Detect which multi-face pixels require full overlap resolution.
│   └── calls _compute_pair_positive_area_overlap_mask(first_polygon_verts=first_pair_polygon_verts, first_polygon_vertex_counts=first_pair_polygon_vertex_counts, second_polygon_verts=second_pair_polygon_verts, second_polygon_vertex_counts=second_pair_polygon_vertex_counts)
├── def _build_visible_face_pixel_polygons(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, clipped_pixel_indices: torch.Tensor, clipped_face_indices: torch.Tensor, face_inverse_depth_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Build exact visible face-pixel polygons in batched tensor form.
│   ├── calls _pack_face_pixel_polygons_by_pixel(clipped_polygon_verts=clipped_polygon_verts, clipped_polygon_vertex_counts=clipped_polygon_vertex_counts, clipped_pixel_indices=clipped_pixel_indices, clipped_face_indices=clipped_face_indices, face_inverse_depth_coefficients=face_inverse_depth_coefficients)
│   ├── if torch.any(single_face_pixel_mask)
│   │   └── calls _gather_visible_pixel_face_polygons(pixel_polygon_verts=pixel_polygon_verts[single_face_pixel_mask], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[single_face_pixel_mask], pixel_face_indices=pixel_face_indices[single_face_pixel_mask], pixel_face_slot_mask=pixel_face_valid_mask[single_face_pixel_mask])
│   └── calls _build_visible_multi_face_pixel_polygons(pixel_indices=pixel_indices[multi_face_pixel_mask], pixel_polygon_verts=pixel_polygon_verts[multi_face_pixel_mask], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[multi_face_pixel_mask], pixel_face_indices=pixel_face_indices[multi_face_pixel_mask], pixel_face_valid_mask=pixel_face_valid_mask[multi_face_pixel_mask], pixel_inverse_depth_coefficients=pixel_inverse_depth_coefficients[multi_face_pixel_mask])
├── def _build_visible_multi_face_pixel_polygons(pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_indices: torch.Tensor, pixel_face_valid_mask: torch.Tensor, pixel_inverse_depth_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Resolve visible polygons for the multi-face pixels in memory-bounded batches.
│   ├── calls _compute_multi_face_pixel_second_bucket_mask(pixel_polygon_verts=pixel_polygon_verts, pixel_polygon_vertex_counts=pixel_polygon_vertex_counts, pixel_face_valid_mask=pixel_face_valid_mask)
│   ├── if torch.any(first_bucket_mask)
│   │   └── calls _gather_visible_pixel_face_polygons(pixel_polygon_verts=pixel_polygon_verts[first_bucket_mask], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[first_bucket_mask], pixel_face_indices=pixel_face_indices[first_bucket_mask], pixel_face_slot_mask=pixel_face_valid_mask[first_bucket_mask])
│   ├── calls _plan_multi_face_pixel_chunks(face_count_per_pixel=face_count_per_pixel, max_verts_per_polygon=max_verts_per_polygon, target_split_line_budget=TARGET_MULTI_FACE_PIXEL_SPLIT_LINE_BUDGET)
│   └── for (chunk_start, chunk_end) in chunk_ranges
│       ├── calls _build_padded_pixel_split_line_coefficients(pixel_indices=pixel_indices[chunk_start:chunk_end], pixel_polygon_verts=pixel_polygon_verts[chunk_start:chunk_end], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[chunk_start:chunk_end], pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end])
│       ├── calls _build_batched_pixel_cell_polygons(pixel_indices=pixel_indices[chunk_start:chunk_end], pixel_polygon_verts=pixel_polygon_verts[chunk_start:chunk_end], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[chunk_start:chunk_end], pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end], pixel_split_line_coefficients=pixel_split_line_coefficients, pixel_split_line_valid_mask=pixel_split_line_valid_mask)
│       └── calls _assign_visible_faces_to_cells(cell_polygon_verts=cell_polygon_verts, cell_polygon_vertex_counts=cell_polygon_vertex_counts, cell_pixel_indices=cell_pixel_indices, pixel_polygon_verts=pixel_polygon_verts[chunk_start:chunk_end], pixel_polygon_vertex_counts=pixel_polygon_vertex_counts[chunk_start:chunk_end], pixel_face_indices=pixel_face_indices[chunk_start:chunk_end], pixel_face_valid_mask=pixel_face_valid_mask[chunk_start:chunk_end], pixel_inverse_depth_coefficients=pixel_inverse_depth_coefficients[chunk_start:chunk_end])
├── def _deduplicate_padded_pixel_split_lines(pixel_split_line_coefficients: torch.Tensor, pixel_split_line_valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   └── # Deduplicate canonical split lines independently within each pixel.
├── def _build_padded_pixel_split_line_coefficients(pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Build padded polygon-edge split-line tensors for all pixels.
│   └── calls _deduplicate_padded_pixel_split_lines(pixel_split_line_coefficients=edge_line_coefficients, pixel_split_line_valid_mask=edge_valid_mask)
├── def _build_batched_pixel_cell_polygons(pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_valid_mask: torch.Tensor, pixel_split_line_coefficients: torch.Tensor, pixel_split_line_valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Build exact arrangement cells for all pixels in batched tensor form.
│   ├── for split_line_index in range(pixel_split_line_coefficients.shape[1])
│   │   ├── calls _clip_convex_polygons_to_half_plane(polygon_verts=candidate_padded_cell_polygon_verts, polygon_vertex_counts=candidate_cell_polygon_vertex_counts, line_coefficients=candidate_line_coefficients)
│   │   ├── calls _clip_convex_polygons_to_half_plane(polygon_verts=candidate_padded_cell_polygon_verts, polygon_vertex_counts=candidate_cell_polygon_vertex_counts, line_coefficients=-candidate_line_coefficients)
│   │   ├── calls _compute_convex_polygon_areas(polygon_verts=positive_polygon_verts, polygon_vertex_counts=positive_polygon_vertex_counts)
│   │   └── calls _compute_convex_polygon_areas(polygon_verts=negative_polygon_verts, polygon_vertex_counts=negative_polygon_vertex_counts)
│   └── calls _compute_convex_polygon_areas(polygon_verts=cell_polygon_verts, polygon_vertex_counts=cell_polygon_vertex_counts)
├── def _assign_visible_faces_to_cells(cell_polygon_verts: torch.Tensor, cell_polygon_vertex_counts: torch.Tensor, cell_pixel_indices: torch.Tensor, pixel_polygon_verts: torch.Tensor, pixel_polygon_vertex_counts: torch.Tensor, pixel_face_indices: torch.Tensor, pixel_face_valid_mask: torch.Tensor, pixel_inverse_depth_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Assign each batched arrangement cell to its frontmost covering face.
│   └── calls _compute_points_in_convex_polygons(points=cell_centroid.unsqueeze(1).expand(-1, candidate_polygon_verts.shape[1], -1).reshape(-1, 2), polygon_verts=candidate_polygon_verts.reshape(-1, candidate_polygon_verts.shape[2], 2), polygon_vertex_counts=candidate_polygon_vertex_counts.reshape(-1))
├── def _pack_face_pixel_polygons_by_pixel(clipped_polygon_verts: torch.Tensor, clipped_polygon_vertex_counts: torch.Tensor, clipped_pixel_indices: torch.Tensor, clipped_face_indices: torch.Tensor, face_inverse_depth_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   └── # Pack variable-count face-pixel polygons into pixel-major padded tensors.
├── def _compute_face_inverse_depth_coefficients(face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor) -> torch.Tensor
│   └── # Compute affine inverse-depth coefficients over projected face triangles.
├── def _build_face_pixel_intersection_polygons(face_screen_verts: torch.Tensor, image_height: int, image_width: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Build exact face-pixel intersection polygons for all candidate pixels.
│   ├── calls _clip_convex_polygons_to_pixel_squares(polygon_verts=polygon_verts, polygon_vertex_counts=polygon_vertex_counts, pixel_x=pixel_x.to(dtype=torch.float32), pixel_y=pixel_y.to(dtype=torch.float32))
│   └── calls _compute_convex_polygon_areas(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts)
├── def _compute_points_in_convex_polygons(points: torch.Tensor, polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   ├── # Test whether each point lies inside its corresponding convex polygon.
│   └── calls _cross_2d(a=next_verts - current_verts, b=points.reshape(-1, 1, 2) - current_verts)
├── def _compute_convex_polygon_bounds(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   └── # Compute axis-aligned bounds for convex polygons.
├── def _compute_points_near_convex_polygon_boundaries(points: torch.Tensor, polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, squared_distance_threshold: float) -> torch.Tensor
│   └── # Test whether each point lies near its corresponding convex polygon boundary.
├── def _compute_convex_polygon_areas(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   └── # Compute areas of convex polygons.
├── def _camera_verts_to_pixel(verts_camera: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor
│   └── # Project camera-space verts into image pixel coordinates.
├── def _clip_convex_polygons_to_pixel_squares(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Clip convex polygons against their corresponding pixel squares.
│   ├── if torch.all(polygon_vertex_counts == 3)
│   │   └── calls _clip_triangle_polygons_to_pixel_squares(triangle_verts=polygon_verts[:, :3, :].contiguous(), pixel_x=pixel_x, pixel_y=pixel_y, output_vertex_capacity=polygon_verts.shape[1])
│   └── for coefficients in line_coefficients
│       └── calls _clip_convex_polygons_to_half_plane(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts, line_coefficients=coefficients)
├── def _clip_convex_polygons_to_half_plane(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, line_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   └── # Clip convex polygons against one half-plane.
├── def _clip_triangle_polygons_to_pixel_squares(triangle_verts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor, output_vertex_capacity: int) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Clip triangles against pixel squares with exact candidate-point geometry.
│   └── calls _compute_points_in_triangles(points=square_corners, triangle_verts=triangle_verts)
├── def _project_screen_polygons_to_face_uv(polygon_verts: torch.Tensor, face_screen_verts: torch.Tensor, face_vertex_depth: torch.Tensor, face_verts_uvs: torch.Tensor) -> torch.Tensor
│   ├── # Map image-space polygon verts to UV using exact perspective interpolation.
│   ├── calls _cross_2d(a=face_screen_v1 - face_screen_v0, b=face_screen_v2 - face_screen_v0)
│   ├── calls _cross_2d(a=face_screen_v1 - polygon_verts, b=face_screen_v2 - polygon_verts)
│   └── calls _cross_2d(a=face_screen_v2 - polygon_verts, b=face_screen_v0 - polygon_verts)
├── def _cross_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor
│   └── # Compute 2D cross product magnitude.
├── def _compute_points_in_triangles(points: torch.Tensor, triangle_verts: torch.Tensor) -> torch.Tensor
│   ├── # Test whether batched points lie inside their corresponding triangles.
│   ├── calls _cross_2d(a=(triangle_v1 - triangle_v0).expand(-1, point_count, -1), b=points - triangle_v0)
│   ├── calls _cross_2d(a=(triangle_v2 - triangle_v1).expand(-1, point_count, -1), b=points - triangle_v1)
│   └── calls _cross_2d(a=(triangle_v0 - triangle_v2).expand(-1, point_count, -1), b=points - triangle_v2)
├── def _compute_convex_polygon_pixel_square_positive_area_overlap_mask(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor, pixel_x: torch.Tensor, pixel_y: torch.Tensor) -> torch.Tensor
│   ├── # Detect positive-area overlap between convex polygons and pixel squares.
│   ├── calls _compute_convex_polygon_bounds(polygon_verts=polygon_verts, polygon_vertex_counts=polygon_vertex_counts)
│   ├── calls _clip_convex_polygons_to_pixel_squares(polygon_verts=clipped_polygon_input_verts, polygon_vertex_counts=polygon_vertex_counts[bbox_overlap_mask], pixel_x=pixel_x[bbox_overlap_mask], pixel_y=pixel_y[bbox_overlap_mask])
│   └── calls _compute_convex_polygon_areas(polygon_verts=clipped_polygon_verts, polygon_vertex_counts=clipped_polygon_vertex_counts)
├── def _duplicate_wrapped_uv_polygons(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Duplicate UV polygons across the cylindrical wrap boundary when needed.
│   └── calls _compute_convex_polygon_bounds(polygon_verts=uv_polygon_verts, polygon_vertex_counts=uv_polygon_vertex_counts)
├── def _build_uv_polygon_texel_intersections(uv_polygon_verts: torch.Tensor, uv_polygon_vertex_counts: torch.Tensor, texture_size: int) -> torch.Tensor
│   ├── # Build exact UV-polygon to texel-cell intersection indices.
│   ├── calls _compute_convex_polygon_bounds(polygon_verts=polygon_texel_verts, polygon_vertex_counts=uv_polygon_vertex_counts)
│   └── while chunk_start < uv_polygon_verts.shape[0]
│       ├── calls _compute_points_in_convex_polygons(points=pixel_centers, polygon_verts=candidate_polygon_verts, polygon_vertex_counts=candidate_polygon_vertex_counts)
│       ├── calls _compute_points_near_convex_polygon_boundaries(points=pixel_centers, polygon_verts=candidate_polygon_verts, polygon_vertex_counts=candidate_polygon_vertex_counts, squared_distance_threshold=boundary_squared_distance_threshold)
│       └── if torch.any(boundary_candidate_mask)
│           └── if len(boundary_triangle_chunks) > 0
│               └── calls _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts=boundary_triangles, pixel_x=boundary_pixel_x[boundary_triangle_candidate_indices], pixel_y=boundary_pixel_y[boundary_triangle_candidate_indices])
├── def _triangulate_convex_uv_polygons(polygon_verts: torch.Tensor, polygon_vertex_counts: torch.Tensor) -> torch.Tensor
│   └── # Triangulate convex UV polygons into a triangle soup.
├── def _duplicate_wrapped_uv_triangles(uv_triangles: torch.Tensor) -> torch.Tensor
│   └── # Duplicate UV triangles across the cylindrical wrap boundary when needed.
├── def _build_uv_triangle_texel_intersections(uv_triangles: torch.Tensor, texture_size: int) -> torch.Tensor
│   ├── # Build exact UV-triangle to texel-cell intersection indices.
│   └── calls _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts=triangle_texel_verts[repeated_triangle_indices], pixel_x=texel_x.to(dtype=torch.float32) + 0.5, pixel_y=texel_y.to(dtype=torch.float32) + 0.5)
└── def _build_uv_triangle_texel_intersections_v2(uv_triangles: torch.Tensor, texture_size: int) -> torch.Tensor
    ├── # Build approximate step-2 `v2` UV-triangle to texel-cell intersections.
    ├── def _compute_triangle_edge_function_coefficients(triangle_verts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] [local]
    │   └── # Compute oriented triangle edge-function coefficients and thresholds.
    ├── calls _compute_triangle_edge_function_coefficients(triangle_verts=triangle_texel_verts)
    └── if torch.any(boundary_candidate_mask)
        └── while boundary_chunk_start < boundary_candidate_indices.shape[0]
            └── calls _compute_triangle_pixel_square_positive_area_overlap_mask(triangle_verts=triangle_texel_verts[repeated_triangle_indices[boundary_chunk_indices]], pixel_x=pixel_x[boundary_chunk_indices], pixel_y=pixel_y[boundary_chunk_indices])
```

```text
models/three_d/meshes/texture/extract/visibility/texel_visibility_v2.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.point_cloud.camera.project import project_3d_to_2d
├── from data.structures.three_d.point_cloud.camera.transform import world_to_camera_transform
├── from models.three_d.meshes.texture.extract.normal_weights import _compute_f_normals_weights
├── FRONT_DEPTH_GAP_LOG_MAD_MULTIPLIER
├── def compute_f_visibility_mask_v2(verts: torch.Tensor, faces: torch.Tensor, camera: Cameras, image_height: int, image_width: int, uv_rasterization_data: Dict[str, torch.Tensor]) -> torch.Tensor
│   ├── # Compute one-view UV-pixel visibility mask from projected texel centers.
│   ├── calls _map_valid_texels_to_continuous_uv_coords(valid_texel_mask=valid_texel_mask)
│   ├── calls _map_continuous_uv_coords_to_barycentric_coords(continuous_uv_coords=continuous_uv_coords, valid_texel_indices=valid_texel_indices, uv_rasterization_data=uv_rasterization_data)
│   ├── calls _filter_texels_by_face_facing(valid_texel_indices=valid_texel_indices, texel_face_indices=texel_face_indices, barycentric_coords=barycentric_coords, verts=verts, faces=faces, camera=camera)
│   ├── calls _map_barycentric_coords_to_3d_world_coords(barycentric_coords=barycentric_coords, texel_face_indices=texel_face_indices, verts=verts, faces=faces)
│   ├── calls _compute_mesh_diagonal(verts=verts)
│   └── calls _compute_texel_visibility_mask_from_world_coords(world_coords=world_coords, valid_texel_indices=valid_texel_indices, valid_texel_mask=valid_texel_mask, mesh_diagonal=mesh_diagonal, camera=camera, image_height=image_height, image_width=image_width)
├── def _map_valid_texels_to_continuous_uv_coords(valid_texel_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
│   └── # Map valid texel centers to continuous UV coordinates.
├── def _map_continuous_uv_coords_to_barycentric_coords(continuous_uv_coords: torch.Tensor, valid_texel_indices: torch.Tensor, uv_rasterization_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
│   ├── # Map continuous UV coordinates to owning-face barycentric coordinates.
│   ├── calls _wrap_continuous_uv_coords_for_faces(continuous_uv_coords=continuous_uv_coords, face_verts_uvs=face_verts_uvs)
│   └── calls _compute_barycentric_coords_in_uv_faces(continuous_uv_coords=wrapped_continuous_uv_coords, face_verts_uvs=face_verts_uvs)
├── def _filter_texels_by_face_facing(valid_texel_indices: torch.Tensor, texel_face_indices: torch.Tensor, barycentric_coords: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, camera: Cameras) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
│   ├── # Filter texels whose owning mesh face is back-facing in the current view.
│   └── calls _compute_f_normals_weights(mesh=Mesh(verts=verts, faces=faces), camera=camera, weights_cfg={'weights': 'normals'})
├── def _map_barycentric_coords_to_3d_world_coords(barycentric_coords: torch.Tensor, texel_face_indices: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor
│   └── # Map barycentric texel coordinates to world-space mesh points.
├── def _compute_texel_visibility_mask_from_world_coords(world_coords: torch.Tensor, valid_texel_indices: torch.Tensor, valid_texel_mask: torch.Tensor, mesh_diagonal: float, camera: Cameras, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Compute texel visibility by keeping the front depth-prefix per pixel.
│   ├── calls world_to_camera_transform(points=world_coords, extrinsics=camera_single.extrinsics, inplace=False)
│   ├── calls project_3d_to_2d(points=texel_camera_coords, intrinsics=camera_single.intrinsics, inplace=False)
│   └── calls _select_visible_depth_clusters_per_camera_pixel(linear_pixel_indices=visible_linear_pixel_indices, depth=visible_projected_depth, mesh_diagonal=mesh_diagonal)
├── def _wrap_continuous_uv_coords_for_faces(continuous_uv_coords: torch.Tensor, face_verts_uvs: torch.Tensor) -> torch.Tensor
│   └── # Wrap texel-center UV coordinates into the seam-safe face-local chart.
├── def _compute_barycentric_coords_in_uv_faces(continuous_uv_coords: torch.Tensor, face_verts_uvs: torch.Tensor) -> torch.Tensor
│   └── # Compute barycentric coordinates of points inside UV triangles.
├── def _compute_mesh_diagonal(verts: torch.Tensor) -> float
│   └── # Compute the full-mesh diagonal length.
├── def _select_visible_depth_clusters_per_camera_pixel(linear_pixel_indices: torch.Tensor, depth: torch.Tensor, mesh_diagonal: float) -> torch.Tensor
│   ├── # Keep only the first front depth cluster in each pixel stack.
│   ├── calls _sort_depth_stacks_per_camera_pixel(linear_pixel_indices=linear_pixel_indices, depth=depth)
│   └── calls _compute_front_depth_gap_threshold_relative(sorted_depth=sorted_depth, segment_start_mask=segment_start_mask, mesh_diagonal=mesh_diagonal)
├── def _sort_depth_stacks_per_camera_pixel(linear_pixel_indices: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
│   └── # Sort projected texels into per-pixel depth stacks.
└── def _compute_front_depth_gap_threshold_relative(sorted_depth: torch.Tensor, segment_start_mask: torch.Tensor, mesh_diagonal: float) -> float
    └── # Derive the front-depth stopping threshold from the gap distribution.
```

```text
models/three_d/meshes/texture/extract/visibility/vertex_visibility.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.mesh.mesh import Mesh
├── from models.three_d.meshes.texture.extract.camera_geometry import _project_verts_to_image, _render_camera_face_index_buffer
├── def compute_v_visibility_mask(mesh: Mesh, camera: Cameras, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Compute one-view binary visibility mask over verts.
│   ├── calls _project_verts_to_image(verts=mesh.verts, camera=camera, image_height=image_height, image_width=image_width)
│   └── calls _compute_rasterized_visible_vertex_mask(verts_camera=verts_camera, faces=mesh.faces.to(device=mesh.device, dtype=torch.long).contiguous(), intrinsics=camera[0].intrinsics, image_height=image_height, image_width=image_width)
├── def _compute_rasterized_visible_vertex_mask(verts_camera: torch.Tensor, faces: torch.Tensor, intrinsics: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor
│   ├── # Compute rasterized one-view vertex visibility mask.
│   ├── calls _compute_face_front_facing_mask(verts_camera=verts_camera, faces=faces)
│   └── calls _render_camera_face_index_buffer(verts_camera=verts_camera, faces=front_facing_faces, intrinsics=intrinsics, image_height=image_height, image_width=image_width)
└── def _compute_face_front_facing_mask(verts_camera: torch.Tensor, faces: torch.Tensor) -> torch.Tensor
    └── # Compute which camera-space mesh faces are front-facing.
```
