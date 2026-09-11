# NerfStudio Tests Structure

## Tests implementation structure

`tests/data/structures/three_d/nerfstudio/test_convert.py`

```text
test_convert.py
├── import numpy as np
├── import pytest
├── from plyfile import PlyData, PlyElement
├── from data.structures.three_d.nerfstudio.convert import _build_colmap_points
├── def test_a_float_colour_reaches_colmap_on_its_own_range()
│   ├── # A float rgb declares 0 to 1, so it is mapped onto COLMAP's 0-to-255 record rather than cast into it, which would send every mid-grey to black.
│   ├── calls write_ply(filepath, rgb_dtype='f4', rgb_values=0.0, 128/255 and 1.0)  # each the 0-to-1 counterpart of a whole 0-to-255 step, so each converts back exactly
│   ├── calls _build_colmap_points(filepath)
│   └── assert the colours of the three points are 0, 128 and 255
├── def test_a_uint16_colour_reaches_colmap_on_its_own_range()
│   ├── # The case the las path produces: a uint16 colour spans 0 to 65535, and casting it to uint8 would wrap rather than scale.
│   ├── calls write_ply(filepath, rgb_dtype='u2', rgb_values=0, 32896 and 65535)  # each a whole multiple of 257, so each converts back exactly
│   ├── calls _build_colmap_points(filepath)
│   └── assert the colours of the three points are 0, 128 and 255
├── def test_a_colour_off_colmap_s_grid_is_refused()
│   ├── # The load converts a colour only losslessly, so a capture whose colour would round onto COLMAP's 0-to-255 grid aborts rather than being exported as a different colour.
│   ├── calls write_ply(filepath, rgb_dtype='f4', rgb_values=0.0, 0.5 and 1.0)
│   └── with pytest.raises(AssertionError)
│       └── calls _build_colmap_points(filepath)
├── def test_a_double_precision_cloud_narrows_once_and_is_checked()
│   ├── # COLMAP records float32 coordinates, so the narrowing is stated at the load where it is value-checked, not applied silently per point.
│   ├── calls write_ply(filepath, xyz_dtype='f8', a coordinate float32 cannot hold exactly)
│   └── with pytest.raises(AssertionError)
│       └── calls _build_colmap_points(filepath)
├── def test_a_cloud_without_colour_is_refused()
│   ├── # A COLMAP point carries a colour, so a cloud that has none is refused rather than exported with a fabricated one.
│   ├── calls write_ply(filepath, with_rgb=False)
│   └── with pytest.raises(AssertionError)
│       └── calls _build_colmap_points(filepath)
└── def write_ply(filepath, xyz_dtype='f4', rgb_dtype='u1', rgb_values=None, with_rgb=True)
    ├── # Writes a single-element PLY whose coordinate and colour widths the caller names, since this suite's whole subject is which width a column arrives in.
    ├── impls columns = the x, y and z properties in xyz_dtype
    ├── if with_rgb
    │   └── impls columns gains red, green and blue in rgb_dtype, holding rgb_values when given
    ├── calls PlyElement.describe(the rows, 'vertex')
    └── calls PlyData.write(filepath)
```
