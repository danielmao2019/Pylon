# Mesh Data Structure Folder Structure

## Code folder structure

`./data/structures/three_d/mesh/`

```text
mesh/
├── __init__.py        # package API surface (re-exports Mesh + MeshTexture types + free functions)
├── mesh.py            # the Mesh class: geometry + optional MeshTexture
├── conventions.py     # geometry coordinate-frame convention transform (opencv right-handed <-> deep3dfacerecon left-handed, etc.)
├── validate.py        # geometry validators (incl. coordinate-frame convention) + texture<->geometry linkage validation
├── load/              # load subpackage: top-level OBJ-vs-GLB dispatch + per-format loaders + block merging
│   ├── __init__.py    # load API surface (re-exports load_mesh)
│   ├── load.py        # load_mesh API: dispatches an OBJ source vs a GLB file
│   ├── load_obj.py    # OBJ file / mesh-root directory -> Mesh (single or multiple blocks; per-representation)
│   ├── load_glb.py    # GLB file -> Mesh (per-representation); thin glTF<->Mesh adapter over utils.io.glb
│   └── merge.py       # merge multiple mesh blocks; texture-atlas packing
├── save/              # save subpackage: top-level OBJ/PLY/GLB dispatch + per-format writers
│   ├── __init__.py    # save API surface (re-exports save_mesh)
│   ├── save.py        # save_mesh API: dispatches OBJ / PLY / GLB by output format
│   ├── save_obj.py    # Mesh -> OBJ (+ sibling MTL + texture PNG)
│   ├── save_ply.py    # Mesh -> PLY (geometry-only / vertex-color)
│   └── save_glb.py    # Mesh -> GLB
├── convert.py         # interop conversions: PyTorch3D / Open3D / trimesh
└── texture/           # the mesh-texture subpackage
    ├── __init__.py                       # texture API surface
    ├── mesh_texture.py                   # MeshTexture ABC
    ├── mesh_texture_vertex_color.py       # MeshTextureVertexColor (per-vertex RGB)
    ├── mesh_texture_uv_texture_map.py     # MeshTextureUVTextureMap (UV atlas; seam-safe canonical verts_uvs)
    ├── conventions.py                     # UV-origin convention transform (obj <-> top_left)
    ├── canonicalize.py                    # OBJ-style vt layout <-> seam-safe canonical layout (shared by load / save / convert)
    ├── texel_face_map.py                   # nvdiffrast-backed texel -> mesh-face correspondence for UV-textured meshes
    ├── validate_vertex_color.py           # vertex-color representation validation
    └── validate_uv_texture_map.py         # uv-texture-map representation validation (incl. seam-safe layout invariant)
```

## Tests folder structure

```text
tests/data/structures/three_d/mesh/
├── test_convert.py
├── test_load_save_roundtrip.py
└── texture/
    ├── test_conventions.py
    ├── test_mesh_texture_uv_texture_map.py
    ├── test_mesh_texture_vertex_color.py
    └── test_texel_face_map.py
```
