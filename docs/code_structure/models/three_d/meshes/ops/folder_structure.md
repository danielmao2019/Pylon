# `models/three_d/meshes/ops/` folder skeleton

## Code folder structure

```text
ops/
├── __init__.py    # MODELS.THREE_D.MESHES.OPS package API surface.
├── arap.py        # As-rigid-as-possible mesh deformation solver and energy.
├── laplacian.py   # Cotangent-Laplacian, edge weights, neighbor data, and geodesics.
├── linear_system.py  # Sparse weighted-Laplacian system assembly, factorization, and solve.
├── normals.py     # Per-vertex normal computation utilities.
└── topology.py    # Undirected edge extraction from triangle faces.
```

## Tests folder structure

```text
tests/models/three_d/meshes/ops/
└── test_normals.py  # Unit-weighted per-vertex normal computation tests.
```
