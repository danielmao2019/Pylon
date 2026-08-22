# `models/three_d/meshes/ops/` tests skeleton

## Tests implementation structure

`tests/models/three_d/meshes/ops/test_apply_transform.py`

```text
test_apply_transform.py
├── import pytest
├── import torch
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from models.three_d.meshes.ops.apply_transform import apply_transform
├── def test_verts_match_reference_matmul() -> None
│   ├── # transformed verts equal a direct homogeneous matmul of mesh.verts by the transform.
│   ├── calls _make_mesh
│   ├── calls _make_transform
│   ├── impls ones_column — a [V, 1] ones tensor in the verts' own dtype and device
│   ├── impls homogeneous_verts — mesh.verts concatenated with ones_column along dim 1
│   ├── impls reference — homogeneous_verts matmul the transform's transpose, keeping the first three columns
│   ├── calls apply_transform(mesh=mesh, transform=transform)
│   └── calls torch.testing.assert_close  # the transformed verts against reference
├── def test_faces_and_texture_preserved() -> None
│   ├── # the returned Mesh keeps the original faces and texture unchanged.
│   ├── calls _make_mesh
│   ├── calls _make_transform
│   ├── calls apply_transform(mesh=mesh, transform=transform)
│   ├── calls torch.testing.assert_close  # the transformed faces against the input mesh's faces
│   └── assert the transformed mesh's texture is the input mesh's texture object itself
├── def test_rejects_non_4x4_transform() -> None
│   ├── # a transform that is not a [4, 4] matrix raises an assertion.
│   ├── calls _make_mesh
│   ├── impls bad_transform — a float32 [3, 3] identity
│   └── with pytest.raises(AssertionError)
│       └── calls apply_transform(mesh=mesh, transform=bad_transform)
├── def _make_mesh() -> Mesh
│   ├── # Builds the one small textured mesh every test in this file transforms, so all three share a single known geometry.
│   ├── impls verts — a float32 [4, 3] tensor holding the origin and the three unit-axis points
│   ├── impls faces — an int64 [2, 3] tensor of two triangles sharing edge 0-2
│   ├── calls MeshTextureVertexColor(vertex_color=a float32 [4, 3] per-vertex color tensor)
│   ├── calls Mesh(verts=verts, faces=faces, texture=texture)
│   └── return  # that Mesh
└── def _make_transform() -> torch.Tensor
    ├── # Builds the one non-trivial affine transform the tests apply, so no passing result can be explained by an identity.
    ├── impls a float32 [4, 4] matrix mixing rotation, scaling, and translation
    └── return  # that transform
```

`tests/models/three_d/meshes/ops/test_normals.py`

```text
test_normals.py
├── import numpy as np
├── import pytest
├── import torch
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
    ├── # Computes a triangle's unit normal using the function-under-test cross(v0 - v1, v1 - v2) convention.
    ├── assert v0 has shape (3,)
    ├── assert v1 has shape (3,)
    ├── assert v2 has shape (3,)
    ├── impls raw — the cross product of v0 - v1 with v1 - v2
    └── return  # raw divided by its L2 norm, the float64 (3,) unit face normal
```
