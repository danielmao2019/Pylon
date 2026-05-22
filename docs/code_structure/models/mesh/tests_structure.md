# Mesh Texture Extraction Tests Structure

Test skeleton for `tests/models/three_d/meshes/texture/`, scoped to the `extract/` module. Branches mirror the system-under-test structure declared in `code_structure.md`; leaves are individual pytest test functions with a one-line purpose. `def` lines carry full prototypes with declaration-level type annotations.

`tests/models/three_d/meshes/texture/` also holds `test_convert.py`, which covers `texture/convert.py` outside the `extract/` scope and is not included here. For module code structure see `code_structure.md`; for the module folder layout see `folder_structure.md`.

## 1. Folder structure

```text
tests/models/three_d/meshes/texture/
├── test_extract.py
├── test_texel_visibility_v2.py
└── test_vertex_visibility.py
```

## 2. Code structure trees

```text
tests/models/three_d/meshes/texture/test_extract.py
├── def test_build_uv_rasterization_mesh_duplicates_seam_crossing_face() -> None
│   └── # _build_uv_rasterization_mesh splits a cylindrical-seam face into two wrapped UV triangles.
├── def test_build_uv_rasterization_mesh_keeps_non_seam_face_single() -> None
│   └── # _build_uv_rasterization_mesh leaves a non-seam face as one UV triangle.
├── def test_build_camera_uv_interpolation_data_shifts_seam_face_once() -> None
│   └── # _build_camera_uv_interpolation_data shifts one seam face into a single continuous UV chart.
├── def test_vertex_uv_to_clip_uses_rasterizer_buffer_v_mapping() -> None
│   └── # _vertex_uv_to_clip maps small-`v` UV coordinates to negative clip-space `y`.
├── def test_compute_f_visibility_mask_keeps_uv_channel_dimension() -> None
│   └── # compute_f_visibility_mask keeps UV visibility masks in `[1, T, T, 1]` layout.
├── def test_compute_f_visibility_mask_uses_exact_camera_pixel_footprints() -> None
│   └── # compute_f_visibility_mask marks visible texels using exact camera-pixel footprints on a one-pixel image.
├── def test_map_visible_screen_space_polygon_regions_to_uv_preserves_identity_face() -> None
│   └── # _map_visible_screen_space_polygon_regions_to_uv maps a polygon to identical UVs on an identity face.
├── def test_break_visible_uv_polygon_regions_into_triangles_triangulates_quad_fan() -> None
│   └── # _triangulate_convex_uv_polygons triangulates one convex quad into a two-triangle fan.
├── def test_compute_visible_uv_texels_from_uv_polygon_regions_uses_top_down_v_convention() -> None
│   └── # _compute_visible_uv_texels_from_uv_polygon_regions maps small-`v` UV triangles into the top texel rows.
├── def test_compute_f_visibility_mask_recovers_standard_uv_face_near_v_zero() -> None
│   └── # compute_f_visibility_mask recovers most occupied texels for one fully visible standard-UV face (CUDA only).
├── def test_extract_texture_from_images_reuses_single_mesh_across_views(monkeypatch: pytest.MonkeyPatch) -> None
│   └── # extract_texture_from_images reuses one shared mesh for all views when a single mesh is given.
├── def test_extract_texture_from_images_uses_per_view_mesh_geometry(monkeypatch: pytest.MonkeyPatch) -> None
│   └── # extract_texture_from_images uses one mesh per view when a mesh list is given.
├── def test_extract_texture_from_images_rejects_per_view_mesh_count_mismatch() -> None
│   └── # extract_texture_from_images rejects a mesh list whose view count mismatches images and cameras.
├── def test_fuse_uv_texture_observations_returns_image_row_order() -> None
│   └── # _fuse_uv_texture_observations returns fused UV outputs in ordinary image row order.
├── def test_fuse_uv_texture_observations_rejects_out_of_range_default_color() -> None
│   └── # _fuse_uv_texture_observations fails instead of clamping an out-of-range fallback color.
├── def test_fuse_vertex_color_observations_rejects_negative_weights() -> None
│   └── # _fuse_vertex_color_observations fails instead of repairing negative fusion weights.
├── def test_extract_uv_texture_map_from_single_image_returns_image_row_order(monkeypatch: pytest.MonkeyPatch) -> None
│   └── # _extract_uv_texture_map_from_single_image returns one-view UV observations in image row order.
├── def test_extract_texture_from_images_keeps_uv_texture_row_order(monkeypatch: pytest.MonkeyPatch) -> None
│   └── # extract_texture_from_images keeps one-view UV extraction coherent through the public API.
└── def test_extract_texture_from_images_rejects_out_of_range_float_images() -> None
    └── # extract_texture_from_images rejects noncanonical out-of-range float RGB images.
```

```text
tests/models/three_d/meshes/texture/test_texel_visibility_v2.py
├── def _build_one_camera() -> Cameras
│   └── # Build one identity OpenCV CPU camera for the focused v2 visibility tests.
├── def test_compute_f_visibility_mask_v2_maps_texel_centers_through_identity_face() -> None
│   └── # compute_f_visibility_mask_v2 keeps the texel-center pipeline consistent on one identity face.
├── def test_compute_f_visibility_mask_v2_filters_back_facing_face_texels() -> None
│   └── # compute_f_visibility_mask_v2 drops texels whose owning face is back-facing in the view.
├── def test_select_visible_depth_clusters_per_camera_pixel_stops_at_first_large_gap() -> None
│   └── # _select_visible_depth_clusters_per_camera_pixel keeps only the front cluster when no later cluster is larger.
├── def test_select_visible_depth_clusters_per_camera_pixel_rejects_larger_second_cluster() -> None
│   └── # _select_visible_depth_clusters_per_camera_pixel stops at the first large gap even when the later cluster is larger.
├── def test_select_visible_depth_clusters_per_camera_pixel_rejects_smaller_second_cluster() -> None
│   └── # _select_visible_depth_clusters_per_camera_pixel keeps the front prefix when the later cluster is smaller.
├── def test_select_visible_depth_clusters_per_camera_pixel_rejects_equal_second_cluster() -> None
│   └── # _select_visible_depth_clusters_per_camera_pixel rejects a later cluster only equal in size to the front cluster.
├── def test_compute_front_depth_gap_threshold_relative_splits_bimodal_gaps() -> None
│   └── # _compute_front_depth_gap_threshold_relative derives a threshold between small surface and large layer gaps.
└── def test_compute_texel_visibility_mask_from_world_coords_keeps_front_depth_prefix() -> None
    └── # _compute_texel_visibility_mask_from_world_coords keeps the front depth prefix under the frame-level MAD threshold.
```

```text
tests/models/three_d/meshes/texture/test_vertex_visibility.py
├── def _build_one_camera() -> Cameras
│   └── # Build one identity OpenCV CUDA camera for the focused vertex-visibility tests.
├── def test_compute_v_visibility_mask_keeps_some_front_facing_triangle_visibility() -> None
│   └── # compute_v_visibility_mask keeps nonzero visibility when the only face is front-facing.
└── def test_compute_v_visibility_mask_filters_back_facing_triangle_vertices() -> None
    └── # compute_v_visibility_mask drops vertices whose only owning face is back-facing.
```
