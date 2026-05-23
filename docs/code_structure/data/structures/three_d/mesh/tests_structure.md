# Mesh Data Structure Tests Structure

Test skeleton for `tests/data/structures/three_d/mesh/`. Branches mirror the
system-under-test structure declared in `code_structure.md`; leaves are
individual pytest test functions with a one-line purpose. The skeleton fixes
what each test pins and the phase-3 implementation follows it exactly. Grow
leaves as we find regressions, contracts, or spec items worth pinning.

`data/structures/three_d/mesh/` is a pure data-structure library, so there are
unit tests only — no end-to-end / running-app tests. For module code structure
see `code_structure.md`; for the module folder layout see `folder_structure.md`.

## 1. Folder structure

```text
tests/data/structures/three_d/mesh/
├── test_convert.py
└── texture/
    ├── test_conventions.py
    ├── test_mesh_texture_vertex_color.py
    └── test_mesh_texture_uv_texture_map.py
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
└── def test_to_converts_uv_convention
    └── # MeshTextureUVTextureMap.to(convention=...) returns a texture whose verts_uvs is converted to the target UV-origin convention.
```
