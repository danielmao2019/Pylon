# Data Viewer Tests Structure

## 1. Tests structure trees

### Dash displays

`tests/data/viewer/utils/displays/test_dash_display_camera_controls.py`

```text
test_dash_display_camera_controls.py
├── import base64
├── import json
├── import math
├── from typing import Optional, Tuple
├── import pytest
├── import torch
├── from dash import dcc
├── from data.structures.three_d.mesh.mesh import Mesh
├── from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import MeshTextureVertexColor
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import ROLL_LOCKED_GRAPH_ID_TYPE
├── from data.viewer.utils.displays.mesh.dash.core_mesh_display import create_dash_mesh_display
├── from data.viewer.utils.displays.points.dash.core_points_display import create_dash_points_display
├── NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)  # deliberately non-axis-aligned, so no world basis vector stands in for it
├── for component in NON_AXIS_ALIGNED_LOCK_ROLL
│   ├── for value in NON_AXIS_ALIGNED_LOCK_ROLL
│   │   └── impls value * value
│   └── impls component / math.sqrt(sum(value * value for value in NON_AXIS_ALIGNED_LOCK_ROLL))
├── NORMALIZED_LOCK_ROLL = tuple(component / math.sqrt(sum(value * value for value in NON_AXIS_ALIGNED_LOCK_ROLL)) for component in NON_AXIS_ALIGNED_LOCK_ROLL)
├── @pytest.mark.parametrize("display_kind", ["points", "mesh"]) def test_a_display_built_without_lock_roll_renders_the_free_trackball(display_kind: str) -> None
│   ├── # A Plotly display built with no lock_roll renders under the free-roll orbit dragmode with no camera.up and no component id, the default trackball control.
│   ├── calls _build_display(display_kind=display_kind, lock_roll=None)   → display
│   ├── impls scene = display.figure.layout.scene.to_plotly_json()
│   ├── assert scene["dragmode"] == "orbit", "..."
│   ├── assert not ("camera" in scene and "up" in scene["camera"]), "..."
│   ├── assert "id" not in display.to_plotly_json()["props"], "..."
│   └── return
├── @pytest.mark.parametrize("display_kind", ["points", "mesh"]) def test_a_display_hands_lock_roll_to_its_controls_unchanged(display_kind: str) -> None
│   ├── # A Plotly display built with a lock_roll renders under the controls that axis builds: the orbit dragmode, the data aspect, the normalized axis as camera.up, and the roll-locked graph id handing the callback that axis.
│   ├── calls _build_display(display_kind=display_kind, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)   → display
│   ├── impls scene = display.figure.layout.scene.to_plotly_json()
│   ├── assert scene["dragmode"] == "orbit", "..."
│   ├── assert scene["aspectmode"] == "data", "..."
│   ├── impls up = scene["camera"]["up"]
│   ├── assert up == pytest.approx({x, y, z of NORMALIZED_LOCK_ROLL}, abs=1e-12), "..."
│   ├── assert display.id["type"] == ROLL_LOCKED_GRAPH_ID_TYPE, "..."
│   ├── impls graph_axis = json.loads(base64.b64decode(display.id["lock_roll"]))
│   ├── assert graph_axis == pytest.approx(NORMALIZED_LOCK_ROLL, abs=1e-12), "..."
│   └── return
└── def _build_display(display_kind: str, lock_roll: Optional[Tuple[float, float, float]]) -> dcc.Graph
    ├── # Builds one Plotly display of the given kind over hand-built synthetic geometry, handing lock_roll to its factory.
    ├── if display_kind == "points"
    │   ├── calls PointCloud(xyz=a float32 tensor of four hand-placed points)   → point_cloud
    │   └── calls create_dash_points_display(point_cloud=point_cloud, lock_roll=lock_roll)   → display
    ├── else
    │   ├── calls MeshTextureVertexColor(vertex_color=a float32 tensor of four RGB colors)   → texture
    │   ├── calls Mesh(verts=the same four points, faces=four hand-placed int64 triangles, texture=texture)   → mesh
    │   └── calls create_dash_mesh_display(mesh=mesh, lock_roll=lock_roll)   → display
    └── return display
```
