# Data Viewer Tests Structure

## Camera controls

### Dash

`tests/data/viewer/utils/controls/camera/camera_controls/dash/test_trackball_camera_controls.py`

```text
test_trackball_camera_controls.py
├── import pytest
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import assert_dash_roll_lock, create_dash_trackball_camera_controls
├── def test_no_axis_renders_no_camera_configuration
│   ├── # A caller that names no lock_roll gets the empty configuration, identical to an explicit lock_roll=None construction.
│   ├── calls create_dash_trackball_camera_controls
│   ├── impls assert the constructed controls carry the same rotation wiring as an explicitly lock_roll=None construction
│   ├── impls assert the constructed controls are the empty configuration, so the display renders the camera it rendered before this argument existed
│   └── return
├── def test_a_supplied_axis_is_held_through_a_drag
│   ├── # Constructing with a lock_roll keeps the camera right axis perpendicular to that supplied axis through a left-drag.
│   ├── calls create_dash_trackball_camera_controls
│   ├── impls construct with a deliberately non-axis-aligned lock_roll
│   ├── impls assert the controls hold the camera right axis perpendicular to that supplied axis at every view direction the drag reaches
│   ├── impls assert the camera up vector coincides with the supplied axis only where the view direction is itself perpendicular to it
│   ├── impls assert the left-drag rotation resolves as yaw about the supplied axis plus pitch about the camera right axis
│   └── return
├── def test_an_applied_camera_state_leaves_the_axis_alone
│   ├── # The lock axis is the caller's, so no CameraState applied by the display or by camera_sync can change what the control locks about.
│   ├── calls create_dash_trackball_camera_controls
│   ├── impls apply a CameraState whose camera up vector differs from the supplied lock_roll
│   ├── impls assert the controls still hold the camera right axis perpendicular to the supplied lock_roll
│   └── return
├── def test_roll_locked_controls_keep_every_other_degree_of_freedom_free
│   ├── # Roll lock constrains roll alone, so a supplied lock_roll construction still passes the mouse-mapping, no-orbit, and no-pose-clamp contracts.
│   ├── calls create_dash_trackball_camera_controls
│   ├── impls assert the construction raises nothing, so its polar angle, azimuth angle, target lock, distance bounds, pan, and translation stay unrestricted
│   └── return
├── def test_assert_dash_roll_lock_rejects_an_ignored_flag
│   ├── # A supplied-lock_roll control whose camera right axis may tilt off perpendicular is rejected, so the flag cannot be silently dropped.
│   ├── impls build a stub control that reports a lock_roll with a camera right axis free to tilt
│   ├── with pytest.raises on the roll-locked-must-keep-the-right-axis-perpendicular-to-the-supplied-axis message
│   │   └── calls assert_dash_roll_lock
│   └── return
└── def test_assert_dash_roll_lock_rejects_an_unrequested_lock
    ├── # A lock_roll=None control that nonetheless holds its camera right axis perpendicular is rejected, so the default construction cannot quietly become roll-locked.
    ├── impls build a stub control that reports lock_roll=None with a camera right axis held perpendicular to a supplied axis
    ├── with pytest.raises on the free-trackball-must-leave-roll-unconstrained message
    │   └── calls assert_dash_roll_lock
    └── return
```

`tests/data/viewer/utils/displays/test_dash_display_camera_controls.py`

```text
test_dash_display_camera_controls.py
├── import pytest
├── from data.viewer.utils.displays.mesh.dash.core_mesh_display import create_dash_mesh_display
├── from data.viewer.utils.displays.points.dash.core_points_display import create_dash_points_display
├── def test_no_axis_renders_no_camera_configuration
│   ├── # Naming no lock_roll leaves the rendered figure carrying no camera configuration, so an existing caller renders what it rendered before the argument existed.
│   ├── calls create_dash_mesh_display
│   ├── calls create_dash_points_display
│   ├── impls assert each returned figure's layout carries no scene configuration
│   └── return
├── def test_a_supplied_axis_renders_a_roll_locked_camera
│   ├── # Naming an axis puts that axis on the rendered camera, so the caller's axis reaches the figure rather than a default the module chose.
│   ├── calls create_dash_mesh_display
│   ├── calls create_dash_points_display
│   ├── impls assert each figure's scene carries the roll-locked dragmode and the normalized supplied axis as camera.up
│   └── return
└── def test_the_two_roll_settings_render_different_cameras
    ├── # The two settings render different cameras, so a passing suite cannot come from the argument being inert.
    ├── calls create_dash_mesh_display
    ├── calls create_dash_points_display
    ├── impls assert the no-axis and supplied-axis scenes differ for each display factory
    └── return
```

### Frontend

```text
frontend roll-lock expectations  # agent-conducted manual procedures, each conducted by driving a running spatial display and looking at the result
├── A display built with no lockRoll left-drag-rotates exactly as it does today, the horizon tilting freely as the drag continues.
├── A display built with a lockRoll holds its horizon level through the same left-drag, the view answering the drag as yaw plus pitch alone.
├── A roll-locked display left-dragged straight up keeps pitching through the pole on its supplied axis and out the far side, the horizon level again on the way down.
├── A roll-locked display right-drag-pans and wheel-zooms over the same range its free counterpart reaches.
└── Two camera-synced displays built with the same lockRoll setting stay in the same pose through a left-drag on either one.
```
