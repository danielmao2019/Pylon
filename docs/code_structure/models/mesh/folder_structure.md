# Mesh Texture Extraction Folder Structure

## 1. Folder structure trees

`./models/three_d/meshes/texture/extract/`

```text
extract/
├── __init__.py             # package API surface (re-exports extract / camera_geometry / visibility functions)
├── extract.py              # main entry: extract_texture_from_images + per-view UV extraction helpers (consumes data-layer build_texel_face_map)
├── camera_geometry.py      # camera-space geometry: world->camera, clip-space, depth- and face-index-buffer rendering
├── normal_weights.py       # normal-alignment weighting helpers
├── weights_cfg.py          # weight-config validation / normalization helpers
└── visibility/             # texel- and vertex-visibility subpackage
    ├── __init__.py                    # visibility API surface
    ├── texel_visibility.py            # exact-UV-polygon texel visibility: compute_f_visibility_mask
    ├── texel_visibility_v2.py         # texel-center-projection texel visibility: compute_f_visibility_mask_v2
    ├── texel_visibility_geometry.py   # low-level texel-visibility geometry kernels
    └── vertex_visibility.py           # vertex visibility: compute_v_visibility_mask
```

The test-folder layout is owned by `tests_structure.md`.
