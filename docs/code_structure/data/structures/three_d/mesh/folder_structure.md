# Mesh Data Structure Folder Structure

## 1. Folder structure trees

`./data/structures/three_d/mesh/`

```text
mesh/
├── __init__.py        # package API surface (re-exports Mesh + free functions)
├── mesh.py            # the Mesh class: geometry + optional texture attributes
├── validate.py        # per-attribute and whole-mesh attribute validators
├── conventions.py     # UV-origin convention transform (obj <-> top_left)
├── load.py            # OBJ file / mesh-root directory -> Mesh constructor kwargs
├── save.py            # Mesh -> OBJ / PLY / MTL / PNG assets on disk
├── merge.py           # merge multiple mesh blocks; texture-atlas packing
└── convert.py         # interop conversions: PyTorch3D / Open3D / trimesh
```

`./tests/data/structures/three_d/mesh/`

```text
mesh/
├── test_conventions.py   # transform_vertex_uv_convention coverage
└── test_convert.py       # PyTorch3D / Open3D / trimesh round-trip coverage
```
