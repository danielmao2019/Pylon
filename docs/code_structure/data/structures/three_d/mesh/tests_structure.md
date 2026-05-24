# Mesh Data Structure Tests Structure

Test skeleton for `tests/data/structures/three_d/mesh/`. Branches mirror the system-under-test structure declared in `code_structure.md`; leaves are individual pytest test functions with a one-line purpose. The skeleton fixes what each test pins and the phase-3 implementation follows it exactly. Grow leaves as we find regressions, contracts, or spec items worth pinning.

`data/structures/three_d/mesh/` is a pure data-structure library, so there are unit tests only — no end-to-end / running-app tests. For module code structure see `code_structure.md`; for the module folder layout see `folder_structure.md`.

## 1. Folder structure

```text
tests/data/structures/three_d/mesh/
├── test_convert.py
├── test_load_save_roundtrip.py
└── texture/
    ├── test_conventions.py
    ├── test_mesh_texture_uv_texture_map.py
    ├── test_mesh_texture_vertex_color.py
    └── test_texel_face_map.py
```

## 2. Code structure trees

```text
tests/data/structures/three_d/mesh/test_convert.py
├── def test_mesh_from_trimesh_welds_seam_to_geometry_domain
│   └── # A seamed UV mesh that trimesh loads in per-corner-expanded form (V == U) must come through mesh_from_trimesh on the canonical geometry domain (V <= U, distinct positions), with the seam carried only by verts_uvs / faces_uvs. Enforces the seam contract (task.md design.2).
├── def test_vertex_count_is_loader_independent
│   └── # For one OBJ asset, len(mesh.verts) must be identical whether the mesh is loaded via Mesh.load (PyTorch3D) or via mesh_from_trimesh, since both land on the canonical geometry domain.
├── def test_trimesh_uv_round_trip_preserves_geometry
│   └── # mesh_to_trimesh then mesh_from_trimesh must preserve geometry, UV, and texture (expand then weld is identity on the geometry domain).
├── def test_pytorch3d_round_trip_preserves_texture
│   └── # mesh_to_pytorch3d then mesh_from_pytorch3d must preserve geometry and texture for both vertex-colored and UV-textured meshes.
└── def test_open3d_round_trip_preserves_vertex_color
    └── # mesh_to_open3d then mesh_from_open3d must preserve geometry and vertex colors (the Open3D path carries no UV texture).
```

```text
tests/data/structures/three_d/mesh/texture/test_conventions.py
├── def test_identity_when_conventions_match
│   └── # transform_verts_uvs_convention returns the UV table unchanged when the source and target conventions are equal.
└── def test_flips_v_axis_when_conventions_differ
    └── # transform_verts_uvs_convention flips the V axis (v -> 1 - v) when the source and target conventions differ.
```

```text
tests/data/structures/three_d/mesh/texture/test_mesh_texture_vertex_color.py
├── def test_normalizes_uint8_to_float01
│   └── # MeshTextureVertexColor normalizes a uint8 [0,255] vertex_color into contiguous float32 [V,3] in [0,1].
├── def test_rejects_out_of_range_float
│   └── # MeshTextureVertexColor rejects a float32 vertex_color carrying values outside [0,1].
└── def test_to_rejects_non_none_convention
    └── # MeshTextureVertexColor.to raises when given a non-None convention, since vertex color carries no UV convention.
```

```text
tests/data/structures/three_d/mesh/texture/test_mesh_texture_uv_texture_map.py
├── def test_rejects_faces_uvs_index_out_of_range
│   └── # MeshTextureUVTextureMap rejects faces_uvs whose indices do not reference valid verts_uvs rows (the cross-field invariant).
├── def test_normalizes_uint8_texture_map
│   └── # MeshTextureUVTextureMap normalizes a uint8 uv_texture_map into contiguous float32 HWC in [0,1].
├── def test_accepts_seam_safe_verts_uvs_outside_unit_interval
│   └── # MeshTextureUVTextureMap accepts verts_uvs whose u extends beyond 1.0 when the per-face u-span stays <= 0.5 (seam-safe canonical form).
├── def test_rejects_face_with_u_span_exceeding_half
│   └── # MeshTextureUVTextureMap rejects any face whose verts_uvs[faces_uvs[f]] u-span exceeds 0.5 (would mean the face straddles the cylindrical wrap without being seam-shifted).
└── def test_to_converts_uv_convention
    └── # MeshTextureUVTextureMap.to(convention=...) returns a texture whose verts_uvs is converted to the target UV-origin convention.
```

```text
tests/data/structures/three_d/mesh/texture/test_texel_face_map.py
├── def test_build_texel_face_map_returns_texel_face_index_and_barycentric
│   └── # build_texel_face_map returns texel_face_index [T, T] int64 and texel_face_barycentric [T, T, 3] float32 with the expected shapes and -1 / NaN sentinels at unoccupied texels.
├── def test_build_texel_face_map_maps_identity_face_to_top_row
│   └── # On one identity-UV face with small-v corners, the returned texel_face_index assigns face 0 to the top texel rows (top_left v-convention is the rasterizer-buffer mapping).
├── def test_build_texel_face_map_covers_both_sides_of_cylindrical_seam
│   └── # For a seam-safe canonical mesh whose only face spans u in {0.95, 1.05, 1.02}, both the u-near-1 and u-near-0 texel columns get assigned to that face (cylindrical wrap coverage via internal seam-side duplication).
└── def test_build_texel_face_map_barycentric_recovers_face_vertex_attributes
    └── # gather(face_attr[faces[texel_face_index]] * texel_face_barycentric).sum(...) recovers a per-vertex attribute on every occupied texel within numerical tolerance.
```

```text
tests/data/structures/three_d/mesh/test_load_save_roundtrip.py
├── def test_load_save_obj_with_seam_face_is_byte_identical
│   └── # Load a hand-written seamed UV OBJ, save it back, and assert byte equality of the resulting vt / f lines — exercises the seam-shift-at-load + collapse-on-save round-trip.
├── def test_load_promotes_seam_crossing_face_to_seam_safe_canonical
│   └── # After load, every face of a seamed mesh satisfies u-span <= 0.5 over its verts_uvs[faces_uvs[f]] (seam-safe canonical form holds).
└── def test_save_collapses_seam_shifted_uv_rows
    └── # collapse_seam_shifted_uv_rows reduces canonical (U' > U) back to OBJ vt structure (U_obj == U): seam-shifted siblings at (u, v) and (u - 1, v) emit one vt line referenced by both face-corner indices.
```
