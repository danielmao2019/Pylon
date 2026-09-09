# `models/three_d/point_cloud/` tests skeleton

## Tests implementation structure

`tests/models/three_d/point_cloud/test_scene_model.py`

```text
test_scene_model.py
├── import pytest
├── import torch
├── from plyfile import PlyData, PlyElement
├── from models.three_d.point_cloud.scene_model import PointCloudSceneModel
├── def test_a_self_describing_file_opens_with_nothing_supplied()
│   ├── # A single-element ply names its own columns, so a scene model opens it with no meta data of any kind.
│   ├── calls write_ply(filepath)
│   ├── calls PointCloudSceneModel.parse_scene_path(filepath)
│   └── assert the model built over it loads and carries xyz
├── def test_a_format_that_names_none_of_its_columns_is_refused_at_the_path()
│   ├── # A .pth defines no layout and nothing here can state one, so it is refused where the path is named rather than aborting inside the loader.
│   ├── calls torch.save(a [N, 3] float32 tensor, filepath)
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloudSceneModel.parse_scene_path(filepath)
├── def test_a_double_precision_file_keeps_its_width_into_the_display()
│   ├── # No load forces a coordinate width here either, so an f8 file reaches the display as float64.
│   ├── calls write_float64_ply(filepath)
│   ├── impls model = the scene model built over that path
│   └── assert the positions it extracts are float64
├── def write_ply(filepath, num_points=8)
│   ├── # Writes a single-element PLY with f4 coordinates, since this suite needs a file that names its own columns.
│   ├── impls vertices = num_points rows of x, y and z as float32
│   └── calls PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)
└── def write_float64_ply(filepath, num_points=8)
    ├── # Writes the same file with f8 coordinates, so the width reaching the display is the file's own.
    ├── impls vertices = num_points rows of x, y and z as float64
    └── calls PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)
```
