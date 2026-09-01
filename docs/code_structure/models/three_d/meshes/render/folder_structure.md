# `models/three_d/meshes/render/` folder skeleton

## Code folder structure

```text
models/three_d/meshes/render/
├── __init__.py      # MODELS.THREE_D.MESHES.RENDER package API surface.
├── core.py          # PyTorch3D RGB/mask rendering from a triangle Mesh and Camera.
├── core_blender.py  # Blender-based RGB/mask rendering parallel to the PyTorch3D stack.
├── display.py       # Scene-model display rendering with snapshot caching and camera overlays.
├── shading.py       # Shading over surface normals, at the band count the coefficients carry.
└── uv_texture.py    # nvdiffrast UV-textured mesh rendering in the renderer's aligned image space.
```

## Tests folder structure

```text
tests/models/three_d/meshes/render/
├── test_core.py          # render_soft_mask_from_mesh and the camera it renders through: the coverage the silhouette carries a gradient through, and both camera models reaching PyTorch3D in PyTorch3D's own frames
├── test_core_blender.py  # render_rgb_from_mesh_blender's camera helper: the Blender camera parameters built from resolution-scaled intrinsics, against a stubbed bpy
└── test_shading.py       # compute_sh_shading: band-count handling and the shading it evaluates over surface normals
```
