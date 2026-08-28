# Mesh Data Structure Tests Structure

## 1. Tests implementation structure

`tests/data/structures/three_d/mesh/test_convert.py`

```text
test_convert.py
├── from data.structures.three_d.mesh.convert import mesh_from_open3d, mesh_from_pytorch3d, mesh_from_trimesh, mesh_to_open3d, mesh_to_pytorch3d, mesh_to_trimesh
├── from data.structures.three_d.mesh.mesh import Mesh
├── def test_mesh_from_trimesh_welds_seam_to_geometry_domain
│   ├── # A seamed UV mesh that trimesh loads in per-corner-expanded form (V == U) must come through mesh_from_trimesh on the canonical geometry domain (V <= U, distinct positions), with the seam carried only by verts_uvs / faces_uvs.
│   ├── calls mesh_from_trimesh
│   ├── impls assert len(mesh.verts) < len(mesh.texture.verts_uvs) for the per-corner-expanded seamed input
│   ├── impls assert every verts row holds a distinct position
│   ├── impls assert the two seam corners at one position index the same verts row
│   ├── impls assert those two corners index different verts_uvs rows
│   └── return
├── def test_vertex_count_is_loader_independent
│   ├── # For one OBJ asset, len(mesh.verts) must be identical whether the mesh is loaded via Mesh.load (PyTorch3D) or via mesh_from_trimesh, since both land on the canonical geometry domain.
│   ├── calls Mesh.load
│   ├── impls trimesh_source = the same OBJ asset loaded through trimesh
│   ├── calls mesh_from_trimesh
│   ├── impls assert both meshes report the same len(mesh.verts)
│   └── return
├── def test_trimesh_uv_round_trip_preserves_geometry
│   ├── # mesh_to_trimesh then mesh_from_trimesh must preserve geometry, UV, and texture (expand then weld is identity on the geometry domain).
│   ├── calls mesh_to_trimesh
│   ├── calls mesh_from_trimesh
│   ├── impls assert the round-tripped verts and faces equal the source mesh's             # impls-node-one-step:skip
│   ├── impls assert the round-tripped verts_uvs and faces_uvs equal the source texture's  # impls-node-one-step:skip
│   ├── impls assert the round-tripped uv_texture_map equals the source texture map
│   └── return
├── def test_pytorch3d_round_trip_preserves_texture
│   ├── # mesh_to_pytorch3d then mesh_from_pytorch3d must preserve geometry and texture for both vertex-colored and UV-textured meshes.
│   ├── for each of a vertex-colored and a UV-textured source mesh
│   │   ├── calls mesh_to_pytorch3d
│   │   ├── calls mesh_from_pytorch3d
│   │   ├── impls assert the round-tripped verts and faces equal that source mesh's  # impls-node-one-step:skip
│   │   └── impls assert the round-tripped texture tensors equal that source texture's
│   └── return
└── def test_open3d_round_trip_preserves_vertex_color
    ├── # mesh_to_open3d then mesh_from_open3d must preserve geometry and vertex colors (the Open3D path carries vertex-color texture only).
    ├── calls mesh_to_open3d
    ├── calls mesh_from_open3d
    ├── impls assert the round-tripped verts and faces equal the source mesh's  # impls-node-one-step:skip
    ├── impls assert the round-tripped vertex_color equals the source vertex color
    └── return
```

`tests/data/structures/three_d/mesh/texture/test_conventions.py`

```text
test_conventions.py
├── from data.structures.three_d.mesh.texture.conventions import transform_uv_convention
├── def test_identity_when_conventions_match
│   ├── # transform_uv_convention returns the UV table unchanged when the source and target conventions are equal.
│   ├── calls transform_uv_convention
│   ├── impls assert the returned UV table equals the input table entry for entry
│   └── return
└── def test_flips_v_axis_when_conventions_differ
    ├── # transform_uv_convention flips the V axis (v -> 1 - v) when the source and target conventions differ.
    ├── calls transform_uv_convention
    ├── impls assert the returned u column equals the input u column
    ├── impls assert the returned v column equals 1 - the input v column
    └── return
```

`tests/data/structures/three_d/mesh/texture/test_mesh_texture_vertex_color.py`

```text
test_mesh_texture_vertex_color.py
├── import pytest
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── def test_normalizes_uint8_to_float01
│   ├── # MeshTextureVertexColor normalizes a uint8 [0,255] vertex_color into contiguous float32 [V,3] in [0,1].
│   ├── calls MeshTextureVertexColor
│   ├── impls assert texture.vertex_color is contiguous float32 [V, 3]
│   ├── impls assert a 255 input channel becomes 1.0, a 0 input channel becomes 0.0
│   └── return
├── def test_rejects_out_of_range_float
│   ├── # MeshTextureVertexColor rejects a float32 vertex_color carrying values outside [0,1].
│   ├── with pytest.raises(AssertionError)
│   │   └── calls MeshTextureVertexColor
│   └── return
└── def test_to_rejects_non_none_uv_convention
    ├── # MeshTextureVertexColor.to raises for a provided UV-origin convention value, since vertex color uses the None UV-convention sentinel.
    ├── calls MeshTextureVertexColor
    ├── with pytest.raises(AssertionError)
    │   └── calls texture.to(uv_convention="obj")
    └── return
```

`tests/data/structures/three_d/mesh/texture/test_mesh_texture_uv_texture_map.py`

```text
test_mesh_texture_uv_texture_map.py
├── import pytest
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── def test_rejects_faces_uvs_index_out_of_range
│   ├── # MeshTextureUVTextureMap rejects faces_uvs whose indices do not reference valid verts_uvs rows (the cross-field invariant).
│   ├── with pytest.raises(AssertionError)
│   │   └── calls MeshTextureUVTextureMap(uv_texture_map=a valid map, verts_uvs=valid rows, faces_uvs=one whose corner index equals len(verts_uvs))
│   └── return
├── def test_normalizes_uint8_texture_map
│   ├── # MeshTextureUVTextureMap normalizes a uint8 uv_texture_map into contiguous float32 HWC in [0,1].
│   ├── calls MeshTextureUVTextureMap
│   ├── impls assert texture.uv_texture_map is contiguous float32 HWC
│   ├── impls assert a 255 input channel becomes 1.0, a 0 input channel becomes 0.0
│   └── return
├── def test_accepts_seam_safe_verts_uvs_outside_unit_interval
│   ├── # MeshTextureUVTextureMap accepts verts_uvs whose u extends beyond 1.0 when each face is non-wrapping (its largest cyclic gap is the wraparound gap), the seam-safe canonical form.
│   ├── calls MeshTextureUVTextureMap(verts_uvs=corner u's {0.95, 1.05, 1.02}, faces_uvs=that one face)
│   ├── impls assert texture.verts_uvs keeps the beyond-1.0 u values unchanged
│   └── return
├── def test_accepts_wide_non_wrapping_face
│   ├── # MeshTextureUVTextureMap accepts a wide face whose u-span exceeds 0.5 while its corners remain contiguous (largest cyclic gap is the wraparound gap).
│   ├── calls MeshTextureUVTextureMap(verts_uvs=corner u's {0.293, 0.735, 0.801}, faces_uvs=that one face)
│   ├── impls assert texture.verts_uvs keeps those corner u's unchanged
│   └── return
├── def test_rejects_wrapping_face
│   ├── # MeshTextureUVTextureMap rejects a face whose largest cyclic gap is an interior gap, requiring seam-shifted contiguous canonical form.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls MeshTextureUVTextureMap(verts_uvs=corner u's {0.02, 0.05, 0.97}, faces_uvs=that one face)
│   └── return
└── def test_to_converts_uv_convention
    ├── # MeshTextureUVTextureMap.to(uv_convention=...) returns a texture whose verts_uvs is converted to the target UV-origin convention.
    ├── calls MeshTextureUVTextureMap
    ├── calls texture.to(uv_convention="top_left")
    ├── impls assert converted.uv_convention is the target uv_convention
    ├── impls assert converted.verts_uvs has V coordinates flipped
    ├── impls assert converted.faces_uvs equals the source faces_uvs
    ├── impls assert converted.uv_texture_map equals the source uv_texture_map
    └── return
```

`tests/data/structures/three_d/mesh/texture/test_texel_face_map.py`

```text
test_texel_face_map.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import MeshTextureUVTextureMap
├── from data.structures.three_d.mesh.texture.texel_face_map import build_texel_face_map
├── def test_build_texel_face_map_returns_texel_face_index_and_barycentric
│   ├── # build_texel_face_map returns texel_face_index [T, T] int64 and texel_face_barycentric [T, T, 3] float32 with the expected shapes and -1 / NaN sentinels at unoccupied texels.
│   ├── calls build_texel_face_map
│   ├── impls assert texel_face_index is [T, T] int64
│   ├── impls assert texel_face_barycentric is [T, T, 3] float32
│   ├── impls assert every unoccupied texel carries -1 in texel_face_index, NaN in texel_face_barycentric
│   └── return
├── def test_build_texel_face_map_maps_identity_face_to_top_row
│   ├── # On one identity-UV face with small-v corners, the returned texel_face_index assigns face 0 to the top texel rows (top_left v-convention is the rasterizer-buffer mapping).
│   ├── calls MeshTextureUVTextureMap(verts_uvs=one identity-UV face's small-v corners, faces_uvs=that one face)
│   ├── calls Mesh
│   ├── calls build_texel_face_map
│   ├── impls assert the texels assigned to face 0 all lie in the top rows of texel_face_index
│   └── return
├── def test_build_texel_face_map_covers_both_sides_of_cylindrical_seam
│   ├── # For a seam-safe canonical mesh whose only face spans u in {0.95, 1.05, 1.02}, both the u-near-1 and u-near-0 texel columns get assigned to that face (cylindrical wrap coverage via internal seam-side duplication).
│   ├── calls build_texel_face_map
│   ├── impls assert the u-near-1 texel columns are assigned to face 0
│   ├── impls assert the u-near-0 texel columns are assigned to face 0
│   └── return
└── def test_build_texel_face_map_barycentric_recovers_face_vertex_attributes
    ├── # barycentric-interpolating the owning face's three corner UVs recovers each occupied texel's own center UV within numerical tolerance, catching corner-permuted barycentric weights.
    ├── calls build_texel_face_map
    ├── impls interpolated_uv = the owning face's corner UVs weighted by texel_face_barycentric, summed over the corner axis
    ├── impls assert interpolated_uv matches each occupied texel's own center UV within tolerance
    └── return
```

`tests/data/structures/three_d/mesh/test_load_save_roundtrip.py`

```text
test_load_save_roundtrip.py
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.canonicalize import collapse_seam_shifted_uv_rows, shift_seam_crossing_faces_to_seam_safe
├── def test_load_save_obj_with_seam_face_is_byte_identical
│   ├── # Load a hand-written seamed UV OBJ, save it back, and assert byte equality of the resulting vt / f lines — exercises the seam-shift-at-load + collapse-on-save round-trip.
│   ├── calls Mesh.load(path=a hand-written seamed UV OBJ)
│   ├── calls mesh.save
│   ├── impls assert the saved vt lines are byte-identical to the source OBJ's
│   ├── impls assert the saved f lines are byte-identical to the source OBJ's
│   └── return
├── def test_load_promotes_seam_crossing_face_to_seam_safe_canonical
│   ├── # After load, every face of a seamed mesh is non-wrapping (its largest cyclic gap over verts_uvs[faces_uvs[f]] is the wraparound gap), the seam-safe canonical form.
│   ├── calls Mesh.load
│   ├── for each face of the loaded mesh
│   │   └── impls assert its largest cyclic gap over verts_uvs[faces_uvs[f]] is the wraparound gap
│   └── return
├── def test_a_vt_row_shared_between_a_shifted_and_an_unshifted_face_is_forked
│   ├── # One vt row can be referenced both by a seam-crossing face and by a face that stays put, and the two want different u's out of it, so the row forks and only the shifting corners follow the copy.
│   ├── calls shift_seam_crossing_faces_to_seam_safe
│   ├── impls assert the returned vt table holds one row more than the source
│   ├── impls assert the appended row sits at the shared row's u plus one, at the same v
│   ├── impls assert the seam-crossing face's corner references the appended row
│   ├── impls assert the unshifted face's corner still references the source row
│   └── return
├── def test_a_mesh_with_no_seam_crossing_face_comes_back_unchanged
│   ├── # A table already in the seam-safe chart is handed straight back, so a load preserves its vt rows and face corners exactly.
│   ├── calls shift_seam_crossing_faces_to_seam_safe
│   ├── impls assert the returned vt table equals the source table row for row
│   ├── impls assert the returned faces equal the source faces corner for corner
│   └── return
└── def test_save_collapses_seam_shifted_uv_rows
    ├── # collapse_seam_shifted_uv_rows reduces canonical (U' > U) back to OBJ vt structure (U_obj == U): seam-shifted siblings at (u, v) and (u - 1, v) emit one vt line referenced by both face-corner indices.
    ├── calls collapse_seam_shifted_uv_rows
    ├── impls assert the returned vt table holds U_obj == U rows
    ├── impls assert the sibling rows at (u, v) and (u - 1, v) collapse to one vt entry  # impls-node-one-step:skip
    ├── impls assert both face-corner indices reference that entry
    └── return
```
