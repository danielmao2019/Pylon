# `models/three_d/point_cloud/` tests skeleton

## Tests implementation structure

`tests/models/three_d/point_cloud/test_scene_model.py`

```text
test_scene_model.py
├── import open3d as o3d
├── import pytest
├── import torch
├── from plyfile import PlyData, PlyElement
├── from models.three_d.point_cloud.scene_model import PointCloudSceneModel
├── def test_a_self_describing_file_opens_with_nothing_supplied()
│   ├── # A single-element ply names its own columns, so a scene model opens it with no meta data of any kind.
│   ├── calls write_ply(filepath)
│   ├── calls PointCloudSceneModel.parse_scene_path(filepath)
│   └── assert the model built over it loads and carries xyz
├── def test_a_format_that_names_none_of_its_columns_opens_on_its_leading_three()
│   ├── # A .pth defines no layout of its own, so the scene model states that its leading three columns are the coordinates.
│   ├── calls torch.save(a [N, 5] float32 tensor, filepath)
│   ├── calls PointCloudSceneModel.parse_scene_path(filepath)
│   ├── impls model = the scene model built over that path
│   └── assert the positions it extracts are the tensor's leading three columns
├── def test_a_pcd_opens_on_its_positions_attribute()
│   ├── # A .pcd defines no default layout, so the scene model names the positions attribute as the coordinates.
│   ├── calls o3d.t.io.write_point_cloud(filepath, a tensor point cloud carrying a positions attribute)
│   ├── impls model = the scene model built over that path
│   └── assert the positions it extracts match that attribute
├── def test_an_off_opens_on_its_vertex_block()
│   ├── # The OFF format declares its vertex block to be the point data, so the scene model names those positional columns as the coordinates.
│   ├── impls filepath = an OFF file whose vertex block holds known coordinates
│   ├── impls model = the scene model built over that path
│   └── assert the positions it extracts match those coordinates
├── def test_an_extension_no_reader_owns_is_refused_at_the_path()
│   ├── # A path the loader has no reader for is refused where it is named rather than aborting inside the load.
│   ├── impls filepath = an existing file whose extension is .xyz
│   └── with pytest.raises(AssertionError)
│       └── calls PointCloudSceneModel.parse_scene_path(filepath)
├── def test_a_double_precision_file_reaches_the_display_at_single_precision()
│   ├── # The display is driven at single precision, so the scene model narrows an f8 file's coordinates itself, a load narrowing nothing.
│   ├── calls write_float64_ply(filepath)
│   ├── impls model = the scene model built over that path
│   └── assert the positions it extracts are float32
├── def write_ply(filepath, num_points=8)
│   ├── # Writes a single-element PLY with f4 coordinates, since this suite needs a file that names its own columns.
│   ├── impls vertices = num_points rows of x, y and z as float32
│   └── calls PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)
└── def write_float64_ply(filepath, num_points=8)
    ├── # Writes the same file with f8 coordinates, so the width reaching the display is the file's own.
    ├── impls vertices = num_points rows of x, y and z as float64
    └── calls PlyData([PlyElement.describe(vertices, 'vertex')]).write(filepath)
```
