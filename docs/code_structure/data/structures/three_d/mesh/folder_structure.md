# Mesh Data Structure Folder Structure

## 1. Folder structure trees

`./data/structures/three_d/mesh/`

```text
mesh/
├── __init__.py        # package API surface (re-exports Mesh + MeshTexture types + free functions)
├── mesh.py            # the Mesh class: geometry + optional MeshTexture
├── validate.py        # geometry validators + texture<->geometry linkage validation
├── load.py            # OBJ file / mesh-root directory -> Mesh constructor kwargs
├── save.py            # Mesh -> OBJ / PLY / MTL / PNG assets on disk
├── merge.py           # merge multiple mesh blocks; texture-atlas packing
├── convert.py         # interop conversions: PyTorch3D / Open3D / trimesh
└── texture/           # the mesh-texture subpackage
    ├── __init__.py                       # texture API surface
    ├── mesh_texture.py                   # MeshTexture ABC
    ├── mesh_texture_vertex_color.py       # MeshTextureVertexColor (per-vertex RGB)
    ├── mesh_texture_uv_texture_map.py     # MeshTextureUVTextureMap (UV atlas; seam-safe canonical verts_uvs)
    ├── conventions.py                     # UV-origin convention transform (obj <-> top_left)
    ├── texel_face_map.py                   # nvdiffrast-backed texel -> mesh-face correspondence for UV-textured meshes
    ├── validate_vertex_color.py           # vertex-color representation validation
    └── validate_uv_texture_map.py         # uv-texture-map representation validation (incl. seam-safe layout invariant)
```

The test-folder layout is owned by `tests_structure.md`.
