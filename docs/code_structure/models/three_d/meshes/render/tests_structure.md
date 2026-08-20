# `models/three_d/meshes/render/` tests skeleton

## Tests implementation structure

### tests/models/three_d/meshes/render/test_shading.py

`tests/models/three_d/meshes/render/test_shading.py`

```text
test_shading.py
├── import pytest
├── from models.three_d.meshes.render.shading import compute_sh_shading
├── def test_band_count_selects_the_spherical_harmonic_order
│   ├── # Coefficients of any perfect-square band count evaluate at the order that count implies, so a caller's band count is never assumed.
│   ├── for each perfect-square band count
│   │   ├── calls compute_sh_shading
│   │   └── impls assert the shading has one RGB triple per input normal
│   └── return
├── def test_non_square_band_count_is_rejected
│   ├── # A coefficient set whose band count is not a perfect square names no spherical-harmonic order, so it fails the assertion rather than evaluating.
│   ├── with pytest.raises(AssertionError)
│   │   └── calls compute_sh_shading
│   └── return
└── def test_shading_varies_with_the_normal_direction
    ├── # Two normals facing differently under the same non-constant coefficients receive different shading, so the basis is really evaluated over the normal.
    ├── calls compute_sh_shading
    ├── impls assert the two normals' shading differs
    └── return
```
