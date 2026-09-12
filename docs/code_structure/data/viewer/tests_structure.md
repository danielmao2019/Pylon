# Data Viewer Tests Structure

## Camera controls

### Dash

`tests/data/viewer/utils/controls/camera/camera_controls/dash/test_trackball_camera_controls.py`

```text
test_trackball_camera_controls.py
├── import pytest
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import apply_dash_trackball_camera_controls, assert_dash_no_camera_pose_clamps, assert_dash_roll_lock, create_dash_trackball_camera_controls
├── def test_no_axis_puts_the_display_under_the_free_roll_dragmode
│   ├── # A display put under the controls with no lock_roll runs the free-roll orbit dragmode and pins no camera.up, identical to an explicit lock_roll=None application.
│   ├── calls apply_dash_trackball_camera_controls
│   ├── impls assert the display figure's scene carries dragmode "orbit" and no camera.up
│   ├── impls assert the scene equals the one an explicitly lock_roll=None application leaves
│   └── return
├── def test_no_axis_registers_no_roll_lock_callback
│   ├── # A display put under the controls with no lock_roll gets no clientside callback, so nothing constrains its roll through a drag.
│   ├── calls apply_dash_trackball_camera_controls
│   ├── impls assert the app's callback map stays empty
│   └── return
├── def test_a_supplied_axis_seeds_the_normalized_axis_as_camera_up
│   ├── # A display put under the controls with a lock_roll carries that axis, normalized, as its figure's camera.up under the free-roll orbit dragmode.
│   ├── impls construct with a deliberately non-axis-aligned lock_roll
│   ├── calls apply_dash_trackball_camera_controls
│   ├── impls assert the display figure's scene carries dragmode "orbit" and camera.up equal to the normalized lock_roll
│   └── return
├── def test_a_non_unit_axis_is_normalized
│   ├── # The caller's axis need not be unit length, so the same direction at any length pins the same camera up vector.
│   ├── calls apply_dash_trackball_camera_controls
│   ├── impls assert the same direction supplied at two lengths yields the same normalized camera up vector
│   └── return
├── def test_a_supplied_axis_registers_the_roll_lock_callback_on_the_display
│   ├── # A display put under the controls with a lock_roll gets the roll-lock clientside callback on its own component id, so the lock holds through the drag rather than at its end.
│   ├── calls apply_dash_trackball_camera_controls
│   ├── impls assert exactly one clientside callback is registered, its one input the display id's relayoutData
│   ├── impls assert its inline source is the shipped roll_lock.js called on the display id and the normalized lock_roll
│   └── return
├── def test_apply_rejects_a_display_without_an_id
│   ├── # A display carrying no component id is rejected, since the roll-lock callback could address no graph.
│   ├── with pytest.raises on the display-must-carry-a-component-id message
│   │   └── calls apply_dash_trackball_camera_controls
│   └── return
├── def test_apply_rejects_a_zero_axis
│   ├── # A zero-length lock_roll names no direction, so it is rejected rather than normalized into a NaN camera.up.
│   ├── with pytest.raises on the non-zero-3-tuple message
│   │   └── calls apply_dash_trackball_camera_controls
│   └── return
├── def test_roll_locked_controls_keep_every_other_degree_of_freedom_free
│   ├── # Roll lock constrains roll alone, so a supplied lock_roll configuration still passes the mouse-mapping, no-orbit, and no-pose-clamp contracts.
│   ├── calls create_dash_trackball_camera_controls
│   ├── impls assert the construction raises nothing, so its azimuth angle, target lock, distance bounds, pan, and translation stay unrestricted
│   └── return
├── def test_the_threejs_viewer_source_passes_the_trackball_contract
│   ├── # The shipped three.js mesh viewer source, handed over the way the mesh display hands it, satisfies every trackball contract and comes back unchanged, so the display's guard keeps guarding it.
│   ├── impls read the shipped renderer JavaScript source
│   ├── calls create_dash_trackball_camera_controls
│   ├── impls assert the returned controls are that source unchanged
│   └── return
├── def test_free_trackball_source_leaves_camera_roll_unconstrained
│   ├── # Renderer source whose left-drag rotation carries the camera up vector passes the free-trackball contract and fails the roll-locked one.
│   ├── impls build renderer source whose left-drag carries the camera up vector
│   ├── calls create_dash_trackball_camera_controls
│   ├── with pytest.raises when the same source is asserted against a supplied axis
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_the_roll_lock_source_holds_the_camera_right_axis_and_up_vector
│   ├── # The shipped roll_lock.js source passes the roll-locked contract and fails the free-trackball one.
│   ├── impls read the shipped roll_lock.js source
│   ├── calls assert_dash_roll_lock
│   ├── with pytest.raises when the same source is asserted with no axis supplied
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_an_ignored_flag
│   ├── # A supplied-lock_roll configuration that pins no camera.up is rejected, so the flag cannot be silently dropped.
│   ├── impls build a free-roll configuration carrying no camera.up
│   ├── with pytest.raises on the roll-locked-must-keep-the-right-axis-perpendicular-to-the-supplied-axis message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_a_mismatched_axis
│   ├── # A configuration pinned to a different axis than the caller supplied is rejected, so the caller's axis cannot be swapped for another.
│   ├── impls build a configuration whose pinned axis differs from the supplied lock_roll
│   ├── with pytest.raises on the roll-locked-must-keep-the-right-axis-perpendicular-to-the-supplied-axis message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_a_source_without_the_polar_band
│   ├── # Roll-locked renderer source that holds the camera right axis perpendicular but bands no polar angle short of the poles is rejected, so a drag through a pole cannot hang the scene upside down.
│   ├── impls build renderer source that re-derives the camera right axis each drag step and carries no polar band
│   ├── with pytest.raises on the roll-locked-must-keep-the-camera-up-vector-on-the-supplied-axis's-side message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_an_unrequested_lock
│   ├── # A lock_roll=None configuration that nonetheless pins camera.up is rejected, so the default construction cannot quietly become roll-locked.
│   ├── impls build a configuration that reports lock_roll=None with a pinned camera.up
│   ├── with pytest.raises on the free-trackball-must-leave-roll-unconstrained message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_no_camera_pose_clamps_rejects_the_pose_clamping_dragmode
│   ├── # The turntable dragmode pins camera.up onto world +Z, so it is rejected as a pose clamp whether or not an axis is supplied.
│   ├── impls build a configuration carrying the turntable dragmode
│   ├── with pytest.raises on the restricted-camera-pose-controls message
│   │   └── calls assert_dash_no_camera_pose_clamps
│   └── return
└── def test_assert_dash_no_camera_pose_clamps_rejects_an_omitted_dragmode
    ├── # A configuration naming no dragmode runs Plotly's turntable default, so it is rejected the same way.
    ├── impls build a configuration carrying no dragmode
    ├── with pytest.raises on the restricted-camera-pose-controls message
    │   └── calls assert_dash_no_camera_pose_clamps
    └── return
```

### Frontend

```text
frontend roll-lock expectations  # agent-conducted manual procedures, each conducted on both stacks by driving a running spatial display in a real browser and reading the camera the page reports
├── A display put under the controls with no lock axis left-drag-rotates as a free trackball, the horizon tilting freely as the drag continues.
├── A display put under the controls with a lock axis holds its horizon level at every pointer move of the same left-drag, the camera right axis perpendicular to the supplied axis.
├── A roll-locked display answers a left-drag as yaw about the supplied axis plus pitch about the camera right axis, a pure-horizontal drag holding its polar angle off the axis and a pure-vertical drag holding its azimuth about it.
├── A roll-locked display's camera up coincides with the supplied axis only where its view direction is perpendicular to that axis.
├── A roll-locked display left-dragged straight up stops pitching at the pole on its supplied axis, the horizon level and the scene upright throughout, and a drag back down turns it off the pole again.
├── A roll-locked display renders no frame rolled off the lock through a long left-drag started inside its viewport, at any drag speed.
├── A roll-locked display's left-drag turns its camera as far per pixel as its free counterpart's.
├── A camera or rotation target written directly onto a roll-locked display — a Plotly.relayout, a Dash figure update with or without a projection switch in it, a modebar reset-camera, a rotation-mode switch in either direction, or a projection switch on Dash; a controls.target or camera.position write on TS — renders no frame off the lock, and on Dash leaves the camera the layout stores on the lock.
├── A camera written onto a roll-locked display with its up on the far side of the axis keeps the eye it was written with, turned upright about its own view direction.
├── A roll-locked display right-drag-pans and wheel-zooms over the same range its free counterpart reaches.
└── Two camera-synced displays built with the same lockRoll setting end every left-drag on either one in the same pose, both roll-locked throughout.
```
