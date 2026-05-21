# Mesh Data Structure Tests Structure

Test-layout skeleton for `tests/data/structures/three_d/mesh/` — the test suite
covering `data/structures/three_d/mesh/`. For the module layout see
`folder_structure.md`; for the module code structure see `code_structure.md`.

## 1. Test folder structure tree

`./tests/data/structures/three_d/mesh/`

```text
mesh/
├── test_conventions.py   # transform_vertex_uv_convention coverage
└── test_convert.py       # mesh_from_*/mesh_to_* round-trip coverage (PyTorch3D / Open3D / trimesh)
```

## 2. Coverage map

```text
data/structures/three_d/mesh/conventions.py   <- tests/data/structures/three_d/mesh/test_conventions.py
data/structures/three_d/mesh/convert.py       <- tests/data/structures/three_d/mesh/test_convert.py
data/structures/three_d/mesh/mesh.py          <- (no dedicated test module)
data/structures/three_d/mesh/validate.py      <- (no dedicated test module)
data/structures/three_d/mesh/load.py          <- (no dedicated test module)
data/structures/three_d/mesh/save.py          <- (no dedicated test module)
data/structures/three_d/mesh/merge.py         <- (no dedicated test module)
```
