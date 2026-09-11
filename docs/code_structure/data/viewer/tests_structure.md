# Viewer Tests Structure

## Tests structure trees

`tests/data/viewer/utils/displays/point_cloud_display/test_point_cloud_display.py`

```text
test_point_cloud_display.py
├── import numpy as np
├── import torch
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.displays.points.dash.core_points_display import create_dash_points_scene
├── def test_a_uint16_colour_is_read_over_its_own_range
│   ├── # The convention is read off what the field MEANS, not off the tensor torch had to widen it into, which is the defect this design exists to close.
│   ├── impls pc = a cloud whose rgb entered as uint16 and is therefore stored in int32
│   ├── calls create_dash_points_scene(point_cloud=pc)
│   └── impls assert the scene's colours span 0 to 255, rescaled from uint16's range rather than int32's
├── def test_a_float_colour_is_read_as_zero_to_one
│   ├── # A float dtype declares the 0-to-1 convention, so no rescale is guessed from the values.
│   ├── impls pc = a cloud whose rgb is float32 with values across 0 to 1
│   ├── calls create_dash_points_scene(point_cloud=pc)
│   └── impls assert the scene's colours are those values scaled to 0 to 255
└── def test_a_uint8_colour_is_passed_through
    ├── # uint8 already spans the display's own range, so the conversion is the identity rather than a second rescale.
    ├── impls pc = a cloud whose rgb is uint8
    ├── calls create_dash_points_scene(point_cloud=pc)
    └── impls assert the scene's colours equal the field's own values
```

```text
the ts point cloud display's colour range
├── # Conducted by loading the display and reading the rendered colours, since the ts frontend carries no unit-test runner.
├── a uchar ply colour renders at the shades the file holds, not white  # THREE reads a Float32 colour attribute over 0 to 1, so a 0-to-255 value would clamp to white at every point
├── a ushort ply colour renders at the same shades as the uchar file holding the matching values  # the source range is uint16's own, which is the case a value-guessing rescale gets wrong
└── a float ply colour renders unchanged  # a float ply type already declares 0 to 1, so nothing is scaled twice
```

`tests/data/viewer/utils/displays/segmentation_display/test_segmentation_pc_colorization.py`

```text
test_segmentation_pc_colorization.py
├── import numpy as np
├── from plyfile import PlyData, PlyElement
├── from data.viewer.utils.displays.points.ts.backend.apis import _map_segmentation_pc_to_rgb
├── def test_a_cloud_with_no_colour_is_written_under_the_ply_column_names
│   ├── # The ply writer names an in-memory coordinate block's and colour block's columns by ply's default layout, so the resource states no layout of its own.
│   ├── impls filepath = a written point cloud carrying class ids and no rgb
│   ├── calls _map_segmentation_pc_to_rgb(segmentation_pc_path=filepath, class_id_to_rgb=a two-class mapping)
│   └── impls assert the written file carries x, y, z, red, green and blue columns
├── def test_a_cloud_that_already_had_colour_is_recoloured_on_the_class_map_s_own_range
│   ├── # The class colours are 0 to 255 whatever the source file declared its own colour to be, and a fresh cloud is what keeps the source's convention from being claimed over them.
│   ├── impls filepath = a written point cloud carrying class ids and a uint16 rgb the file named
│   ├── calls _map_segmentation_pc_to_rgb(segmentation_pc_path=filepath, class_id_to_rgb=a two-class mapping)
│   └── impls assert the written colour columns are u1 holding the mapping's own values
└── def test_the_written_cloud_carries_the_class_colours_and_nothing_else
    ├── # The resource exists to be displayed, so the source's remaining columns reach no output column.
    ├── impls filepath = a written point cloud carrying class ids and an intensity column
    ├── calls _map_segmentation_pc_to_rgb(segmentation_pc_path=filepath, class_id_to_rgb=a two-class mapping)
    └── impls assert the written file carries no intensity column
```
