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
│   ├── impls ones_column — a [V, 1] ones tensor in the verts' own dtype and device  # impls-node-one-step:skip
│   ├── impls homogeneous_verts — mesh.verts concatenated with ones_column along dim 1
│   ├── impls reference — homogeneous_verts matmul the transform's transpose, keeping the first three columns
│   ├── calls apply_transform(mesh=mesh, transform=transform)
│   └── calls torch.testing.assert_close(transformed.verts, reference)
├── def test_faces_and_texture_preserved() -> None
│   ├── # the returned Mesh keeps the original faces and texture unchanged.
│   ├── calls _make_mesh
│   ├── calls _make_transform
│   ├── calls apply_transform(mesh=mesh, transform=transform)
│   ├── calls torch.testing.assert_close(transformed.faces, mesh.faces)
│   └── assert the transformed mesh's texture is the input mesh's texture object itself
├── def test_rejects_non_4x4_transform() -> None
│   ├── # a transform that is not a [4, 4] matrix raises an assertion.
│   ├── calls _make_mesh
│   ├── impls bad_transform — a float32 [3, 3] identity
│   └── with pytest.raises(AssertionError)
│       └── calls apply_transform(mesh=mesh, transform=bad_transform)
├── def _make_mesh() -> Mesh
│   ├── # Builds the one small textured mesh every test in this file transforms, so all three share a single known geometry.
│   ├── impls verts — a float32 [4, 3] tensor holding the origin and the three unit-axis points  # impls-node-one-step:skip
│   ├── impls faces — an int64 [2, 3] tensor of two triangles sharing edge 0-2
│   ├── calls MeshTextureVertexColor(vertex_color=a float32 [4, 3] per-vertex color tensor)
│   ├── calls Mesh(verts=verts, faces=faces, texture=texture)
│   └── return  # that Mesh
└── def _make_transform() -> torch.Tensor
    ├── # Builds the one non-trivial affine transform the tests apply, so no passing result can be explained by an identity.
    ├── impls a float32 [4, 4] matrix mixing rotation, scaling, and translation  # impls-node-one-step:skip
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
│   ├── impls verts — a float32 [4, 3] tensor of four non-coplanar positions
│   ├── impls faces — an int64 [2, 3] tensor of two triangles sharing edge 0-1
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
│   ├── impls norms — the L2 norm of each returned normal along the last axis
│   └── calls torch.testing.assert_close(norms, torch.ones_like(norms))
├── def test_single_planar_triangle_orientation() -> None
│   ├── # A single z=0 planar triangle yields the (0, 0, +1) unit normal at all verts.
│   ├── impls verts — a float32 [3, 3] tensor of one z=0 triangle
│   ├── impls faces — an int64 [1, 3] tensor naming that single triangle
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
│   ├── impls expected — a float32 [3, 3] tensor of (0, 0, 1) repeated once per vertex
│   └── calls torch.testing.assert_close(normals, expected)
├── def test_unit_weighting_not_area_weighting() -> None
│   ├── # A shared-edge tent verifies unit weighting, not area weighting, of face normals.
│   ├── impls a, b, c, d — four float64 (3,) positions forming a non-coplanar tent over shared edge A-B
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=c)  # -> n0
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=d)  # -> n1
│   ├── impls raw0 — the un-normalized cross product of a - b with b - c
│   ├── impls raw1 — the un-normalized cross product of a - b with b - d
│   ├── impls area0 — half the L2 norm of raw0
│   ├── impls area1 — half the L2 norm of raw1
│   ├── assert area0 and area1 are not close, so the two weightings cannot coincide
│   ├── impls unit_weighted — n0 + n1, L2-normalized
│   ├── impls area_weighted — area0 * n0 + area1 * n1, L2-normalized
│   ├── assert unit_weighted and area_weighted are not all-close
│   ├── impls verts — the four positions stacked into a float32 [4, 3] tensor
│   ├── impls faces — an int64 [2, 3] tensor of the two tent triangles
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
│   ├── impls expected_unit — unit_weighted as a float32 tensor
│   ├── impls expected_area — area_weighted as a float32 tensor
│   ├── calls torch.testing.assert_close(normals[0], expected_unit)
│   ├── assert the shared vertex's normal is not all-close to expected_area, at atol 1e-4 and rtol 0
│   ├── calls torch.testing.assert_close(normals[2], torch.tensor(n0, dtype=torch.float32))
│   └── calls torch.testing.assert_close(normals[3], torch.tensor(n1, dtype=torch.float32))
├── def test_batched_matches_unbatched() -> None
│   ├── # Each batch element's result equals the corresponding single-mesh call.
│   ├── impls verts — a float32 [4, 3] tensor
│   ├── impls faces — an int64 [2, 3] tensor of two triangles sharing edge 0-1
│   ├── impls verts_other — verts scaled by two and shifted by one             # impls-node-one-step:skip
│   ├── impls batched — verts and verts_other stacked into a [2, 4, 3] tensor  # impls-node-one-step:skip
│   ├── calls compute_vertex_normals(verts=batched, faces=faces, weights="unit")
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="unit")
│   ├── calls compute_vertex_normals(verts=verts_other, faces=faces, weights="unit")
│   ├── assert the batched result has shape (2, 4, 3)
│   ├── calls torch.testing.assert_close(normals_batched[0], normals_first)
│   └── calls torch.testing.assert_close(normals_batched[1], normals_second)
├── def test_area_output_is_unit_length() -> None
│   ├── # Every weights="area" per-vertex normal is L2-normalized on a non-degenerate mesh.
│   ├── impls verts — a float32 [4, 3] tensor of four non-coplanar positions
│   ├── impls faces — an int64 [2, 3] tensor of two triangles sharing edge 0-1
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="area")
│   ├── impls norms — the L2 norm of each returned normal along the last axis
│   └── calls torch.testing.assert_close(norms, torch.ones_like(norms))
├── def test_area_weighting_not_unit_weighting() -> None
│   ├── # A shared-edge tent verifies weights="area" applies area, not unit, weighting of face normals.
│   ├── impls a, b, c, d — four float64 (3,) positions forming a non-coplanar tent over shared edge A-B
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=c)  # -> n0
│   ├── calls _face_unit_normal(v0=a, v1=b, v2=d)  # -> n1
│   ├── impls raw0 — the un-normalized cross product of a - b with b - c
│   ├── impls raw1 — the un-normalized cross product of a - b with b - d
│   ├── impls area0 — half the L2 norm of raw0
│   ├── impls area1 — half the L2 norm of raw1
│   ├── assert area0 and area1 are not close, so the two weightings cannot coincide
│   ├── impls unit_weighted — n0 + n1, L2-normalized
│   ├── impls area_weighted — area0 * n0 + area1 * n1, L2-normalized
│   ├── assert unit_weighted and area_weighted are not all-close
│   ├── impls verts — the four positions stacked into a float32 [4, 3] tensor
│   ├── impls faces — an int64 [2, 3] tensor of the two tent triangles
│   ├── calls compute_vertex_normals(verts=verts, faces=faces, weights="area")
│   ├── impls expected_unit — unit_weighted as a float32 tensor
│   ├── impls expected_area — area_weighted as a float32 tensor
│   ├── calls torch.testing.assert_close(normals[0], expected_area)
│   ├── assert the shared vertex's normal is not all-close to expected_unit, at atol 1e-4 and rtol 0
│   ├── calls torch.testing.assert_close(normals[2], torch.tensor(n0, dtype=torch.float32))
│   └── calls torch.testing.assert_close(normals[3], torch.tensor(n1, dtype=torch.float32))
├── def test_unrecognized_weights_trips_dispatch_assert() -> None
│   ├── # An unrecognized weights value trips the dispatch fall-through assert.
│   ├── impls verts — a float32 [3, 3] tensor of one triangle
│   ├── impls faces — an int64 [1, 3] tensor naming that single triangle
│   └── with pytest.raises(AssertionError)
│       └── calls compute_vertex_normals(verts=verts, faces=faces, weights="bogus")
└── def _face_unit_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray
    ├── # Computes a triangle's unit normal using the function-under-test cross(v0 - v1, v1 - v2) convention.
    ├── assert v0 has shape (3,)
    ├── assert v1 has shape (3,)
    ├── assert v2 has shape (3,)
    ├── impls raw — the cross product of v0 - v1 with v1 - v2
    └── return  # raw divided by its L2 norm, the float64 (3,) unit face normal
```
