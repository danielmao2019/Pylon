# `models/three_d/meshes/ops/` tests skeleton

## Tests implementation structure

`tests/models/three_d/meshes/ops/test_apply_transform.py`

```text
test_apply_transform.py
├── def test_verts_match_reference_matmul
│   └── # transformed verts equal a direct homogeneous matmul of mesh.verts by the transform.
├── def test_faces_and_texture_preserved
│   └── # the returned Mesh keeps the original faces and texture unchanged.
└── def test_rejects_non_4x4_transform
    └── # a transform that is not a [4, 4] matrix raises an assertion.
```

`tests/models/three_d/meshes/ops/test_normals.py`

```text
test_normals.py
├── from models.three_d.meshes.ops import compute_vertex_normals
├── def test_output_is_unit_length() -> None
│   ├── # Every returned per-vertex normal is L2-normalized on a non-degenerate mesh.
│   └── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
├── def test_single_planar_triangle_orientation() -> None
│   ├── # A single z=0 planar triangle yields the (0, 0, +1) unit normal at all verts.
│   └── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
├── def test_unit_weighting_not_area_weighting() -> None
│   ├── # A shared-edge tent verifies unit weighting, not area weighting, of face normals.
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=c)
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=d)
│   └── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
├── def test_batched_matches_unbatched() -> None
│   ├── # Each batch element's result equals the corresponding single-mesh call.
│   ├── calls compute_vertex_normals(verts=batched, faces=faces, weights="unit")
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
│   └── calls compute_vertex_normals(verts=verts_other, faces=faces, weights="unit")
├── def test_area_output_is_unit_length() -> None
│   ├── # Every weights="area" per-vertex normal is L2-normalized on a non-degenerate mesh.
│   └── calls compute_vertex_normals(verts=verts, faces=faces, weights="area")
├── def test_area_weighting_not_unit_weighting() -> None
│   ├── # A shared-edge tent verifies weights="area" applies area, not unit, weighting of face normals.
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=c)
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=d)
│   └── calls compute_vertex_normals(verts=verts, faces=faces, weights="area")
├── def test_unrecognized_weights_trips_dispatch_assert() -> None
│   ├── # An unrecognized weights value trips the dispatch fall-through assert.
│   └── calls compute_vertex_normals(verts=verts, faces=faces, weights="bogus")
└── def _face_unit_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray
    └── # Computes a triangle's unit normal using the function-under-test cross(v0 - v1, v1 - v2) convention.
```
