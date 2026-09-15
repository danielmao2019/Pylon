# Data Viewer Tests Structure

## 1. Tests structure trees

### Dash displays

`tests/data/viewer/utils/displays/test_dash_display_camera_controls.py`

```text
test_dash_display_camera_controls.py
├── import base64
├── import json
├── import pytest
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import ROLL_LOCKED_GRAPH_ID_TYPE
├── from data.viewer.utils.displays.mesh.dash.core_mesh_display import create_dash_mesh_display
├── from data.viewer.utils.displays.points.dash.core_points_display import create_dash_points_display
├── def test_a_display_built_without_lock_roll_renders_the_free_trackball(display_kind)  # @pytest.mark.parametrize("display_kind", ["points", "mesh"])
│   ├── # A Plotly display built with no lock_roll renders under the free-roll orbit dragmode with no camera.up and no component id, the default trackball control.
│   ├── calls _build_display(display_kind=display_kind, lock_roll=None)
│   ├── impls assert the figure's layout.scene dragmode is "orbit", its camera carries no up, and the dcc.Graph carries no id
│   └── return
├── def test_a_display_hands_lock_roll_to_its_controls_unchanged(display_kind)  # @pytest.mark.parametrize("display_kind", ["points", "mesh"])
│   ├── # A Plotly display built with a lock_roll renders under the controls that axis builds: the orbit dragmode, the data aspect, the normalized axis as camera.up, and the roll-locked graph id handing the callback that axis.
│   ├── impls construct with a deliberately non-axis-aligned lock_roll
│   ├── calls _build_display(display_kind=display_kind, lock_roll=lock_roll)
│   ├── impls assert the figure's layout.scene carries dragmode "orbit", aspectmode "data", and camera.up equal to the normalized lock_roll
│   ├── impls assert the dcc.Graph id's type is ROLL_LOCKED_GRAPH_ID_TYPE and json.loads(base64.b64decode(its lock_roll)) equals the normalized lock_roll
│   └── return
└── def _build_display(display_kind, lock_roll)
    ├── # Builds one Plotly display of the given kind over hand-built synthetic geometry, handing lock_roll to its factory.
    ├── if display_kind == "points"
    │   ├── impls point_cloud = a PointCloud of a few hand-placed points
    │   └── calls create_dash_points_display(point_cloud=point_cloud, lock_roll=lock_roll)
    ├── else
    │   ├── impls mesh = a vertex-colored Mesh of a few hand-placed triangles
    │   └── calls create_dash_mesh_display(mesh=mesh, lock_roll=lock_roll)
    └── return display
```
