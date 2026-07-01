# `models/three_d/meshes/ops/` folder skeleton

## Code folder structure

```text
ops/
├── __init__.py                     # MODELS.THREE_D.MESHES.OPS package API surface.
├── apply_transform.py              # maps a Mesh's verts through a 4x4 transform via the chunked large×small matmul
├── arap.py                         # As-rigid-as-possible mesh deformation solver and energy.
├── laplacian.py                    # Cotangent-Laplacian, edge weights, neighbor data, and geodesics.
├── linear_system.py                # Sparse weighted-Laplacian system assembly, factorization, and solve.
├── normals.py                      # Per-vertex normal computation utilities.
├── topology.py                     # Undirected edge extraction from triangle faces.
└── world_to_camera_transform.py    # high-level world-to-camera mesh transform, implemented via apply_transform
```

## Tests folder structure

```text
tests/models/three_d/meshes/ops/
├── test_apply_transform.py  # transformed verts match a reference homogeneous matmul; faces and texture preserved
└── test_normals.py          # Unit-weighted per-vertex normal computation tests.
```
