# `data/viewer/utils/displays/points/` tests skeleton

## Tests implementation structure

`tests/data/viewer/utils/displays/point_cloud_display/test_dash_points_style_args.py`

```text
test_dash_points_style_args.py
├── import numpy as np
├── import plotly.graph_objects as go
├── import pytest
├── import torch
├── from dash import dcc
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── from data.viewer.utils.displays.points.dash.core_points_display import DEFAULT_POINT_COLOR, DEFAULT_POINT_SIZE_FLOOR, DEFAULT_POINT_SIZE_RATIO, create_dash_points_component, create_dash_points_display, create_dash_points_scene
├── @pytest.fixture def large_radius_xyz()
│   ├── # Synthetic xyz whose bounding-sphere radius is large enough that the size heuristic beats the floor.
│   ├── impls corner = 1000.0
│   ├── impls coords = []
│   ├── for sx in (-1.0, 1.0)
│   │   └── for sy in (-1.0, 1.0)
│   │       └── for sz in (-1.0, 1.0)
│   │           └── impls coords.append([sx * corner, sy * corner, sz * corner])
│   ├── impls xyz = those eight cube corners as a float32 tensor
│   └── return xyz
├── @pytest.fixture def tiny_radius_xyz()
│   ├── # Synthetic xyz whose bounding-sphere radius is so small the size heuristic falls back to the floor.
│   ├── impls corner = 0.001
│   ├── impls coords = []
│   ├── for sx in (-1.0, 1.0)
│   │   └── for sy in (-1.0, 1.0)
│   │       └── for sz in (-1.0, 1.0)
│   │           └── impls coords.append([sx * corner, sy * corner, sz * corner])
│   ├── impls xyz = those eight cube corners as a float32 tensor
│   └── return xyz
├── def test_scene_explicit_point_size_used(large_radius_xyz)
│   ├── # An explicit point_size sets marker.size verbatim.
│   ├── calls PointCloud(xyz=large_radius_xyz)                          # -> pc
│   ├── calls create_dash_points_scene(point_cloud=pc, point_size=7.5)  # -> trace
│   ├── assert the trace is a go.Scatter3d
│   └── assert its marker size is 7.5
├── def test_scene_point_size_falls_back_to_radius_heuristic(large_radius_xyz)
│   ├── # With no point_size, the radius-relative size is used where it beats the floor.
│   ├── calls PointCloud(xyz=large_radius_xyz)    # -> pc
│   ├── calls _expected_radius(large_radius_xyz)  # -> radius
│   ├── impls expected = max(DEFAULT_POINT_SIZE_FLOOR, radius * DEFAULT_POINT_SIZE_RATIO)
│   ├── assert expected beats the floor                                  # guards this test's own premise
│   ├── calls create_dash_points_scene(point_cloud=pc, point_size=None)  # -> trace
│   ├── assert its marker size is approximately expected
│   └── assert its marker size is approximately radius * DEFAULT_POINT_SIZE_RATIO
├── def test_scene_point_size_falls_back_to_floor(tiny_radius_xyz)
│   ├── # With no point_size, the floor is used where the radius-relative size falls below it.
│   ├── calls PointCloud(xyz=tiny_radius_xyz)                            # -> pc
│   ├── calls _expected_radius(tiny_radius_xyz)                          # -> radius
│   ├── assert radius * DEFAULT_POINT_SIZE_RATIO falls below the floor   # guards this test's own premise
│   ├── calls create_dash_points_scene(point_cloud=pc, point_size=None)  # -> trace
│   └── assert its marker size is approximately DEFAULT_POINT_SIZE_FLOOR
├── def test_scene_explicit_point_color_uniform(large_radius_xyz)
│   ├── # An explicit point_color sets one uniform marker color, over the point cloud's own rgb.
│   ├── impls rgb = eight random uint8 triplets
│   ├── calls PointCloud(xyz=large_radius_xyz, data={"rgb": rgb})              # -> pc
│   ├── calls create_dash_points_scene(point_cloud=pc, point_color="#ff0000")  # -> trace
│   └── assert its marker color is "#ff0000"
├── def test_scene_point_color_none_uses_per_point_rgb(large_radius_xyz)
│   ├── # With no point_color, the point cloud's own per-point rgb is used.
│   ├── impls rgb = eight consecutive uint8 triplets
│   ├── calls PointCloud(xyz=large_radius_xyz, data={"rgb": rgb})         # -> pc
│   ├── calls create_dash_points_scene(point_cloud=pc, point_color=None)  # -> trace
│   ├── impls color = trace.marker.color
│   ├── assert color is an np.ndarray
│   ├── assert its shape is (8, 3)
│   └── impls assert color equals rgb entry by entry
├── def test_scene_point_color_none_no_rgb_uses_default(large_radius_xyz)
│   ├── # With no point_color and no rgb, DEFAULT_POINT_COLOR is used.
│   ├── calls PointCloud(xyz=large_radius_xyz)  # -> pc
│   ├── calls pc.field_names()
│   ├── assert the point cloud carries no 'rgb'
│   ├── calls create_dash_points_scene(point_cloud=pc, point_color=None)  # -> trace
│   └── assert its marker color is DEFAULT_POINT_COLOR
├── def test_create_dash_points_display_returns_graph(large_radius_xyz)
│   ├── # create_dash_points_display returns a dcc.Graph wrapping the one scene trace.
│   ├── calls PointCloud(xyz=large_radius_xyz)            # -> pc
│   ├── calls create_dash_points_display(point_cloud=pc)  # -> graph
│   ├── assert the graph is a dcc.Graph
│   ├── assert its figure is a go.Figure
│   ├── assert that figure holds one trace
│   └── assert that trace is a go.Scatter3d
├── def test_create_dash_points_display_passes_style_args(large_radius_xyz)
│   ├── # The style overrides reach the wrapped Scatter3d marker.
│   ├── calls PointCloud(xyz=large_radius_xyz)  # -> pc
│   ├── calls create_dash_points_display(point_cloud=pc, point_size=4.0, point_color="#00ff00")  # -> graph
│   ├── impls trace = graph.figure.data[0]
│   ├── assert its marker size is 4.0
│   └── assert its marker color is "#00ff00"
├── def test_create_dash_points_component_wraps_scene(large_radius_xyz)
│   ├── # create_dash_points_component wraps a Scatter3d into a single-trace Graph under the free trackball its caller builds.
│   ├── calls PointCloud(xyz=large_radius_xyz)                              # -> pc
│   ├── calls create_dash_points_scene(point_cloud=pc, point_size=3.0)      # -> scene
│   ├── calls create_dash_trackball_camera_controls()                       # -> controls
│   ├── calls create_dash_points_component(scene=scene, controls=controls)  # -> graph
│   ├── assert the graph is a dcc.Graph
│   ├── assert its figure is a go.Figure
│   ├── assert that figure holds one trace
│   └── assert that trace's marker size is 3.0
└── def _expected_radius(xyz: torch.Tensor) -> float
    ├── # Reproduces the source's bounding-radius computation, so the size assertions stand on their own.
    ├── impls points_np = xyz.detach().cpu().numpy()
    ├── impls center = points_np.mean(axis=0)
    ├── impls radius = the largest distance from center to a point
    └── return radius
```
