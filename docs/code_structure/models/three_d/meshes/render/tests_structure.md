# `models/three_d/meshes/render/` tests

## Tests implementation structure

### tests/models/three_d/meshes/render/test_core.py

`tests/models/three_d/meshes/render/test_core.py`

```text
test_core.py
├── def test_soft_silhouette_shape_and_range
│   └── # render_soft_silhouette_from_mesh returns an [H, W] float tensor with values in [0, 1].
├── def test_soft_silhouette_is_differentiable
│   └── # The soft silhouette backpropagates non-zero gradients to the mesh vertices.
└── def test_blur_sigma_changes_the_silhouette
    └── # The same mesh rendered under two blur sigmas yields different soft silhouettes.
```
