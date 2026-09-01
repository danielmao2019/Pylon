# `tests/models/three_d/meshes/texture/` tests skeleton

## 1. Tests structure trees

`tests/models/three_d/meshes/texture/test_extract.py`

```text
test_extract.py
├── import pytest
├── import torch
├── from typing import Dict
├── from models.three_d.meshes.texture.extract.extract import extract_texture_from_images, _extract_uv_texture_map_from_single_image, _fuse_uv_texture_observations, _fuse_vertex_color_observations
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility import compute_f_visibility_mask, _compute_visible_uv_texels_from_uv_polygon_regions, _map_visible_screen_space_polygon_regions_to_uv
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility_geometry import triangulate_convex_uv_polygons
├── def test_compute_f_visibility_mask_keeps_uv_channel_dimension() -> None
│   ├── # compute_f_visibility_mask keeps UV visibility masks in `[1, T, T, 1]` layout.
│   ├── calls _build_texel_face_map_stub
│   ├── calls compute_f_visibility_mask(texel_face_map=texel_face_map)
│   ├── impls assert the returned mask's shape is [1, texture_size, texture_size, 1]
│   └── return
├── def test_compute_f_visibility_mask_uses_exact_camera_pixel_footprints() -> None
│   ├── # compute_f_visibility_mask marks visible texels using exact camera-pixel footprints on a one-pixel image.
│   ├── calls _build_texel_face_map_stub
│   ├── calls compute_f_visibility_mask(image_height=1, image_width=1, texel_face_map=texel_face_map)
│   ├── impls assert every texel the one pixel's footprint covers is marked visible
│   ├── impls assert no texel outside that footprint is marked visible
│   └── return
├── def test_map_visible_screen_space_polygon_regions_to_uv_preserves_identity_face() -> None
│   ├── # _map_visible_screen_space_polygon_regions_to_uv maps a polygon to identical UVs on an identity face.
│   ├── calls _map_visible_screen_space_polygon_regions_to_uv(face_verts_uvs=an identity face whose uvs equal its screen verts)
│   ├── impls assert the returned uv polygon verts equal the input screen polygon verts
│   └── return
├── def test_break_visible_uv_polygon_regions_into_triangles_triangulates_quad_fan() -> None
│   ├── # triangulate_convex_uv_polygons triangulates one convex quad into a two-triangle fan.
│   ├── calls triangulate_convex_uv_polygons(polygon_verts=one convex quad, polygon_vertex_counts=a single count of four)
│   ├── impls assert the result is the two triangles of a fan anchored at the quad's first corner
│   └── return
├── def test_compute_visible_uv_texels_from_uv_polygon_regions_uses_top_down_v_convention() -> None
│   ├── # _compute_visible_uv_texels_from_uv_polygon_regions maps small-`v` UV triangles into the top texel rows.
│   ├── calls _compute_visible_uv_texels_from_uv_polygon_regions(uv_polygon_verts=a triangle at small v, texture_size=texture_size)
│   ├── impls assert the marked texels sit in the mask's top rows
│   └── return
├── def test_compute_f_visibility_mask_recovers_standard_uv_face_near_v_zero() -> None
│   ├── # compute_f_visibility_mask recovers most occupied texels for one fully visible standard-UV face (CUDA only).
│   ├── impls skip the test when no CUDA device is available
│   ├── calls _build_texel_face_map_stub
│   ├── calls compute_f_visibility_mask(texel_face_map=texel_face_map)
│   ├── impls assert the marked texel count is most of the face's occupied texels
│   └── return
├── def test_extract_texture_from_images_reuses_single_mesh_across_views(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # extract_texture_from_images reuses one shared mesh for all views when a single mesh is given.
│   ├── impls patch _extract_uv_texture_map_from_single_image to record the mesh it receives per view
│   ├── calls extract_texture_from_images(mesh=one Mesh, images=a multi-view image stack)
│   ├── impls assert every recorded mesh is that same object
│   └── return
├── def test_extract_texture_from_images_uses_per_view_mesh_geometry(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # extract_texture_from_images uses one mesh per view when a mesh list is given.
│   ├── impls patch _extract_uv_texture_map_from_single_image to record the mesh it receives per view
│   ├── calls extract_texture_from_images(mesh=a per-view Mesh list, images=a multi-view image stack)
│   ├── impls assert the recorded meshes are that list, in view order
│   └── return
├── def test_extract_texture_from_images_rejects_per_view_mesh_count_mismatch() -> None
│   ├── # extract_texture_from_images rejects a mesh list whose view count mismatches images and cameras.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls extract_texture_from_images(mesh=a mesh list one shorter than the views)
│   └── return
├── def test_fuse_uv_texture_observations_returns_image_row_order() -> None
│   ├── # _fuse_uv_texture_observations returns fused UV outputs in ordinary image row order.
│   ├── calls _fuse_uv_texture_observations(observations=two hand-built single-view observations)
│   ├── impls assert the fused map's first row carries the observation at the smallest v
│   └── return
├── def test_fuse_uv_texture_observations_rejects_out_of_range_default_color() -> None
│   ├── # _fuse_uv_texture_observations fails instead of clamping an out-of-range fallback color.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls _fuse_uv_texture_observations(default_color=a value outside [0, 1])
│   └── return
├── def test_fuse_vertex_color_observations_rejects_negative_weights() -> None
│   ├── # _fuse_vertex_color_observations fails instead of repairing negative fusion weights.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls _fuse_vertex_color_observations(observations=an observation carrying a negative weight)
│   └── return
├── def test_extract_uv_texture_map_from_single_image_returns_image_row_order(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # _extract_uv_texture_map_from_single_image returns one-view UV observations in image row order.
│   ├── calls _build_texel_face_map_stub
│   ├── calls _extract_uv_texture_map_from_single_image(texel_face_map=texel_face_map)
│   ├── impls assert the observation's first row carries the texel at the smallest v
│   └── return
├── def test_extract_texture_from_images_keeps_uv_texture_row_order(monkeypatch: pytest.MonkeyPatch) -> None
│   ├── # extract_texture_from_images keeps one-view UV extraction coherent through the public API.
│   ├── calls _build_texel_face_map_stub
│   ├── calls extract_texture_from_images(images=a single-view image stack)
│   ├── impls assert the returned texture equals the single-view extraction's own fused map
│   └── return
├── def _build_texel_face_map_stub(texture_size: int) -> Dict[str, torch.Tensor]
│   ├── # Build a uniform fully-occupied texel_face_map assigning every texel to face 0 with centroid barycentrics.
│   ├── impls texel_face_index = a [texture_size, texture_size] tensor of zeros
│   ├── impls texel_barycentrics = a [texture_size, texture_size, 3] tensor of one-third weights
│   ├── impls texel_valid_mask = a [texture_size, texture_size] tensor of True
│   └── return the three tensors under their texel_face_map keys
└── def test_extract_texture_from_images_rejects_out_of_range_float_images() -> None
    ├── # extract_texture_from_images rejects noncanonical out-of-range float RGB images.
    ├── with pytest.raises(AssertionError)
    │   └── calls extract_texture_from_images(images=a float image stack with values outside [0, 1])
    └── return
```

`tests/models/three_d/meshes/texture/test_texel_visibility_v2.py`

```text
test_texel_visibility_v2.py
├── import torch
├── from typing import Dict
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from models.three_d.meshes.texture.extract.visibility.texel_visibility_v2 import compute_f_visibility_mask_v2, _compute_front_depth_gap_threshold_relative, _compute_texel_visibility_mask_from_world_coords, _select_visible_depth_clusters_per_camera_pixel
├── def test_compute_f_visibility_mask_v2_maps_texel_centers_through_identity_face() -> None
│   ├── # compute_f_visibility_mask_v2 keeps the texel-center pipeline consistent on one identity face.
│   ├── calls _build_one_camera
│   ├── calls _build_texel_face_map_with_three_texels
│   ├── calls compute_f_visibility_mask_v2(camera=camera, texel_face_map=texel_face_map)
│   ├── impls assert exactly the three occupied texels are marked visible
│   └── return
├── def test_compute_f_visibility_mask_v2_filters_back_facing_face_texels() -> None
│   ├── # compute_f_visibility_mask_v2 drops texels whose owning face is back-facing in the view.
│   ├── calls _build_one_camera
│   ├── calls _build_texel_face_map_with_three_texels
│   ├── calls compute_f_visibility_mask_v2(camera=camera, texel_face_map=texel_face_map)
│   ├── impls assert no texel of the back-facing face is marked visible
│   └── return
├── def test_select_visible_depth_clusters_per_camera_pixel_stops_at_first_large_gap() -> None
│   ├── # _select_visible_depth_clusters_per_camera_pixel keeps only the front cluster when no later cluster is larger.
│   ├── calls _select_visible_depth_clusters_per_camera_pixel(depth=a front cluster followed by a large gap)
│   ├── impls assert only the front cluster's entries are kept
│   └── return
├── def test_select_visible_depth_clusters_per_camera_pixel_rejects_larger_second_cluster() -> None
│   ├── # _select_visible_depth_clusters_per_camera_pixel stops at the first large gap even when the later cluster is larger.
│   ├── calls _select_visible_depth_clusters_per_camera_pixel(depth=a small front cluster before a larger far cluster)
│   ├── impls assert the far cluster's entries are dropped
│   └── return
├── def test_select_visible_depth_clusters_per_camera_pixel_rejects_smaller_second_cluster() -> None
│   ├── # _select_visible_depth_clusters_per_camera_pixel keeps the front prefix when the later cluster is smaller.
│   ├── calls _select_visible_depth_clusters_per_camera_pixel(depth=a front cluster before a smaller far cluster)
│   ├── impls assert only the front prefix is kept
│   └── return
├── def test_select_visible_depth_clusters_per_camera_pixel_rejects_equal_second_cluster() -> None
│   ├── # _select_visible_depth_clusters_per_camera_pixel rejects a later cluster only equal in size to the front cluster.
│   ├── calls _select_visible_depth_clusters_per_camera_pixel(depth=two equally-sized clusters split by a large gap)
│   ├── impls assert only the front cluster is kept
│   └── return
├── def test_compute_front_depth_gap_threshold_relative_splits_bimodal_gaps() -> None
│   ├── # _compute_front_depth_gap_threshold_relative derives a threshold between small surface and large layer gaps.
│   ├── calls _compute_front_depth_gap_threshold_relative(sorted_depth=a bimodal gap sequence, mesh_diagonal=mesh_diagonal)
│   ├── impls assert the returned threshold sits above every small surface gap
│   ├── impls assert the returned threshold sits below every large layer gap
│   └── return
├── def test_compute_texel_visibility_mask_from_world_coords_keeps_front_depth_prefix() -> None
│   ├── # _compute_texel_visibility_mask_from_world_coords keeps the front depth prefix under the frame-level MAD threshold.
│   ├── calls _build_one_camera
│   ├── calls _compute_texel_visibility_mask_from_world_coords(world_coords=two depth layers behind one pixel, camera=camera)
│   ├── impls assert only the near layer's texels are marked visible
│   └── return
├── def _build_one_camera() -> Cameras
│   ├── # Build one identity OpenCV CPU camera for the focused v2 visibility tests.
│   ├── calls build_camera_intrinsics(model="pinhole", params=that camera's focal lengths, principal point and its own h and w, intr_convention="standard", device="cpu")
│   ├── calls CameraExtrinsics(extrinsics=the identity 4x4, extr_convention="opencv", device="cpu")
│   ├── calls Cameras(intrinsics=[camera_intrinsics], extrinsics=[camera_extrinsics], device="cpu")
│   └── return the built camera  # one identity-extrinsic Cameras
└── def _build_texel_face_map_with_three_texels(face_index: int, occupied_positions: tuple) -> Dict[str, torch.Tensor]
    ├── # Build a [2, 2] texel_face_map assigning the given (row, col) positions to the given face with centroid barycentrics.
    ├── impls texel_face_index = a [2, 2] tensor holding face_index at occupied_positions
    ├── impls texel_barycentrics = a [2, 2, 3] tensor of one-third weights
    ├── impls texel_valid_mask = a [2, 2] tensor true only at occupied_positions
    └── return the three tensors under their texel_face_map keys
```

`tests/models/three_d/meshes/texture/test_vertex_visibility.py`

```text
test_vertex_visibility.py
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.structures.three_d.camera.extrinsics.camera_extrinsics import CameraExtrinsics
├── from data.structures.three_d.camera.intrinsics.camera_intrinsics import build_camera_intrinsics
├── from models.three_d.meshes.texture.extract.visibility.vertex_visibility import compute_v_visibility_mask
├── def test_compute_v_visibility_mask_keeps_some_front_facing_triangle_visibility() -> None
│   ├── # compute_v_visibility_mask keeps nonzero visibility when the only face is front-facing.
│   ├── calls _build_one_camera
│   ├── calls compute_v_visibility_mask(mesh=a single front-facing triangle mesh, camera=camera)
│   ├── impls assert at least one vert is marked visible
│   └── return
├── def test_compute_v_visibility_mask_filters_back_facing_triangle_verts() -> None
│   ├── # compute_v_visibility_mask drops verts whose only owning face is back-facing.
│   ├── calls _build_one_camera
│   ├── calls compute_v_visibility_mask(mesh=a single back-facing triangle mesh, camera=camera)
│   ├── impls assert no vert is marked visible
│   └── return
└── def _build_one_camera() -> Cameras
    ├── # Build one identity OpenCV CUDA camera for the focused vertex-visibility tests.
    ├── calls build_camera_intrinsics(model="pinhole", params=that camera's focal lengths, principal point and its own h and w, intr_convention="standard", device="cuda")
    ├── calls CameraExtrinsics(extrinsics=the identity 4x4, extr_convention="opencv", device="cuda")
    ├── calls Cameras(intrinsics=[camera_intrinsics], extrinsics=[camera_extrinsics], device="cuda")
    └── return the built camera  # one identity-extrinsic Cameras
```
