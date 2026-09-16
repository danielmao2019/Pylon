# Data Viewer Camera Controls Tests Structure

## 1. Tests structure trees

### Dash

`tests/data/viewer/utils/controls/camera/camera_controls/dash/test_trackball_camera_controls.py`

```text
test_trackball_camera_controls.py
├── import base64
├── import json
├── import math
├── import subprocess
├── import sys
├── from pathlib import Path
├── from typing import Optional, Tuple
├── import pytest
├── from dash import ALL, Input
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import ROLL_LOCK_CALLBACK_SCRIPT, ROLL_LOCKED_GRAPH_ID_TYPE, assert_dash_no_camera_pose_clamps, assert_dash_roll_lock, create_dash_trackball_camera_controls
├── from data.viewer.utils.displays.mesh.dash.core_mesh_display import TEXTURED_MESH_VIEWER_SCRIPT_PATH
├── REPO_ROOT = Path(__file__).resolve().parents[8]  # the working dir the registration probe imports the module from
├── NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)  # deliberately non-axis-aligned, so no world basis vector stands in for it
├── NON_UNIT_LOCK_ROLL = (3.0, 9.0, -2.0)  # the same direction at ten times the length
├── for component in NON_AXIS_ALIGNED_LOCK_ROLL
│   ├── for value in NON_AXIS_ALIGNED_LOCK_ROLL
│   │   └── impls value * value
│   └── impls component / math.sqrt(sum(value * value for value in NON_AXIS_ALIGNED_LOCK_ROLL))
├── NORMALIZED_LOCK_ROLL = tuple(component / math.sqrt(sum(value * value for value in NON_AXIS_ALIGNED_LOCK_ROLL)) for component in NON_AXIS_ALIGNED_LOCK_ROLL)
├── FREE_TRACKBALL_RENDERER_SOURCE = renderer source wiring the trackball mouse mapping whose left-drag turns camera.up with the eye
├── REGISTRATION_PROBE_SCRIPT = script a fresh interpreter runs to print, as JSON, the global callbacks on the roll-locked graph pattern and the global inline scripts
├── def test_no_axis_builds_the_free_roll_controls
│   ├── # Plotly controls built with no lock_roll run the free-roll orbit dragmode, pin no camera.up, and carry no graph id, so the roll-lock callback matches nothing of theirs.
│   ├── calls create_dash_trackball_camera_controls()   → controls
│   ├── assert controls == {"scene": {"dragmode": "orbit"}, "graph_id": None}, "..."
│   └── return
├── def test_a_supplied_axis_seeds_the_normalized_axis_as_camera_up
│   ├── # Plotly controls built with a lock_roll carry that axis, normalized, as their scene's camera.up under the free-roll orbit dragmode.
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)   → controls
│   ├── assert controls["scene"]["dragmode"] == "orbit", "..."
│   ├── assert controls["scene"]["aspectmode"] == "data", "..."
│   ├── impls up = controls["scene"]["camera"]["up"]
│   ├── assert up == pytest.approx({x, y, z of NORMALIZED_LOCK_ROLL}, abs=1e-12), "..."
│   └── return
├── def test_a_non_unit_axis_is_normalized
│   ├── # The caller's axis need not be unit length, so the same direction at any length pins the same camera up vector and hands the callback the same axis.
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)   → unit_controls
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_UNIT_LOCK_ROLL)   → scaled_controls
│   ├── impls unit_up, scaled_up = the two controls' scene camera.up
│   ├── assert scaled_up == pytest.approx(unit_up, abs=1e-12), "..."
│   ├── assert math.isclose(math.hypot(scaled_up x, y, z), 1.0), "..."
│   ├── impls unit_graph_axis, scaled_graph_axis = each controls' graph_id lock_roll, base64-decoded then JSON-loaded
│   ├── assert scaled_graph_axis == pytest.approx(unit_graph_axis, abs=1e-12), "..."
│   └── return
├── def test_a_supplied_axis_carries_the_roll_locked_graph_id
│   ├── # Plotly controls built with a lock_roll carry a ROLL_LOCKED_GRAPH_ID_TYPE component id handing the callback the normalized axis, its index unique per construction so two roll-locked graphs on one page never share an id.
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)   → first_controls
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)   → second_controls
│   ├── assert first_controls["graph_id"]["type"] == ROLL_LOCKED_GRAPH_ID_TYPE, "..."
│   ├── impls graph_axis = json.loads(base64.b64decode(first_controls["graph_id"]["lock_roll"]))
│   ├── assert graph_axis == pytest.approx(NORMALIZED_LOCK_ROLL, abs=1e-12), "..."
│   ├── impls graph_id_text = the graph_id values joined
│   ├── assert "." not in graph_id_text, "..."  # Dash escapes "." in output ids
│   ├── assert the two graph_id indices differ, "..."
│   └── return
├── def test_a_supplied_axis_pins_the_data_aspect
│   ├── # Plotly controls built with a lock_roll draw the scene at its data's own proportions, so the world axis keeps its direction in the scene's space, while free controls leave the aspect to Plotly.
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)   → roll_locked_controls
│   ├── calls create_dash_trackball_camera_controls()   → free_controls
│   ├── assert roll_locked_controls["scene"]["aspectmode"] == "data", "..."
│   ├── assert "aspectmode" not in free_controls["scene"], "..."
│   └── return
├── def test_the_roll_lock_callback_is_registered_once_on_the_roll_locked_graph_pattern
│   ├── # Importing the module registers exactly one clientside callback, matching every roll-locked graph's relayoutData by pattern and running the shipped roll_lock.js source, so no display registers a callback of its own.
│   ├── impls completed = subprocess.run([sys.executable, "-c", REGISTRATION_PROBE_SCRIPT], cwd=REPO_ROOT, capture_output=True, text=True)  # a fresh interpreter: the first Dash app to set up its server empties the global callback lists
│   ├── assert completed.returncode == 0, "..."
│   ├── impls probe = json.loads(completed.stdout)
│   ├── assert len(probe["callbacks"]) == 1, "..."
│   ├── impls callback = probe["callbacks"][0]
│   ├── impls expected_input = Input({"type": ROLL_LOCKED_GRAPH_ID_TYPE, "index": ALL, "lock_roll": ALL}, "relayoutData").to_dict()
│   ├── assert callback["inputs"] == [expected_input], "..."
│   ├── impls registering_scripts = an empty list of the inline scripts that name the callback's clientside function
│   ├── for script in probe["inline_scripts"]
│   │   └── if callback["clientside_function"]["function_name"] in script
│   │       └── impls registering_scripts.append(script)
│   ├── assert exactly one registering script, holding ROLL_LOCK_CALLBACK_SCRIPT, "..."
│   └── return
├── def test_a_zero_axis_is_rejected
│   ├── # A zero-length lock_roll names no direction, so it is rejected rather than normalized into a NaN camera.up.
│   ├── with pytest.raises on the non-zero-3-tuple message
│   │   └── calls create_dash_trackball_camera_controls
│   └── return
├── def test_roll_locked_controls_keep_every_other_degree_of_freedom_free
│   ├── # Roll lock constrains roll alone, so roll-locked Plotly controls still pass the mouse-mapping, no-orbit, and no-pose-clamp contracts.
│   ├── calls create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)  # the factory runs every trackball contract, so returning without raising is the assertion
│   └── return
├── def test_the_threejs_viewer_source_passes_the_trackball_contract
│   ├── # The shipped three.js mesh viewer source, handed over the way the mesh display hands it, satisfies every trackball contract and comes back unchanged, so the display's guard keeps guarding it.
│   ├── impls source = the shipped three.js viewer source read from TEXTURED_MESH_VIEWER_SCRIPT_PATH, its __CAMERA_SYNC_SCRIPT__ placeholder stripped as the mesh display does
│   ├── calls create_dash_trackball_camera_controls(renderer_controls=source)   → constructed
│   ├── assert constructed is source, "..."
│   └── return
├── def test_free_trackball_source_leaves_camera_roll_unconstrained
│   ├── # Renderer source whose left-drag rotation carries the camera up vector passes the free-trackball contract and fails the roll-locked one.
│   ├── calls create_dash_trackball_camera_controls(renderer_controls=FREE_TRACKBALL_RENDERER_SOURCE)   → constructed
│   ├── assert constructed is FREE_TRACKBALL_RENDERER_SOURCE, "..."
│   ├── with pytest.raises when the same source is asserted against a supplied axis
│   │   └── calls assert_dash_roll_lock(controls=FREE_TRACKBALL_RENDERER_SOURCE, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
│   └── return
├── def test_the_roll_lock_source_holds_the_camera_right_axis_and_up_vector
│   ├── # The shipped roll_lock.js source passes the roll-locked contract and fails the free-trackball one.
│   ├── calls assert_dash_roll_lock(controls=ROLL_LOCK_CALLBACK_SCRIPT, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
│   ├── with pytest.raises when the same source is asserted with no axis supplied
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_an_ignored_flag
│   ├── # Supplied-lock_roll Plotly controls that pin no camera.up are rejected, so the flag cannot be silently dropped.
│   ├── impls build free-roll Plotly controls carrying no camera.up
│   ├── with pytest.raises on the roll-locked-must-keep-the-right-axis-perpendicular-to-the-supplied-axis message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_a_mismatched_axis
│   ├── # Plotly controls pinned to a different axis than the caller supplied are rejected, so the caller's axis cannot be swapped for another.
│   ├── impls build Plotly controls whose pinned axis differs from the supplied lock_roll
│   ├── with pytest.raises on the roll-locked-must-keep-the-right-axis-perpendicular-to-the-supplied-axis message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_a_graph_id_the_callback_does_not_match
│   ├── # Roll-locked Plotly controls whose graph id is missing, of another type, or carrying another axis are rejected, so the roll-lock callback can neither miss the graph nor hold it about the wrong axis.
│   ├── impls controls = roll-locked Plotly controls built about NORMALIZED_LOCK_ROLL
│   ├── impls variants = copies of controls, one per mismatched graph_id: None, a foreign type, a different lock_roll
│   ├── for each variant
│   │   └── with pytest.raises on the roll-locked-Plotly-controls-must-carry-the-graph-id-the-roll-lock-callback-matches message
│   │       └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_a_scene_not_at_data_proportions
│   ├── # Roll-locked Plotly controls whose scene leaves the aspect to Plotly are rejected, since a stretched scene turns the world axis away from the direction the seeded camera.up names.
│   ├── impls controls = roll-locked Plotly controls built about NORMALIZED_LOCK_ROLL
│   ├── impls variants = copies of controls, one per scene aspect other than data proportions: no aspectmode, aspectmode "cube"
│   ├── for each variant
│   │   └── with pytest.raises on the roll-locked-Plotly-controls-must-draw-the-scene-at-its-data's-own-proportions message
│   │       └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_a_source_without_the_polar_band
│   ├── # Roll-locked renderer source that holds the camera right axis perpendicular but bands no polar angle short of the poles is rejected, so a drag through a pole cannot hang the scene upside down.
│   ├── impls build renderer source that re-derives the camera right axis each drag step and carries no polar band
│   ├── with pytest.raises on the roll-locked-must-keep-the-camera-up-vector-on-the-supplied-axis's-side message
│   │   └── calls assert_dash_roll_lock
│   └── return
├── def test_assert_dash_roll_lock_rejects_an_unrequested_lock
│   ├── # lock_roll=None Plotly controls that nonetheless pin camera.up or carry a roll-locked graph id are rejected, so the default construction cannot quietly become roll-locked.
│   ├── impls build lock_roll=None Plotly controls with a pinned camera.up, and ones with a roll-locked graph_id
│   ├── for each of the two
│   │   └── with pytest.raises on the free-trackball-must-leave-roll-unconstrained message
│   │       └── calls assert_dash_roll_lock
│   └── return
├── @pytest.mark.parametrize("lock_roll", [None, NON_AXIS_ALIGNED_LOCK_ROLL]) def test_assert_dash_no_camera_pose_clamps_rejects_the_pose_clamping_dragmode(lock_roll: Optional[Tuple[float, float, float]]) -> None
│   ├── # The turntable dragmode pins camera.up onto world +Z, so it is rejected as a pose clamp whether or not an axis is supplied.
│   ├── impls build Plotly controls whose scene carries the turntable dragmode
│   ├── with pytest.raises on the restricted-camera-pose-controls message
│   │   └── calls assert_dash_no_camera_pose_clamps
│   └── return
└── @pytest.mark.parametrize("lock_roll", [None, NON_AXIS_ALIGNED_LOCK_ROLL]) def test_assert_dash_no_camera_pose_clamps_rejects_an_omitted_dragmode(lock_roll: Optional[Tuple[float, float, float]]) -> None
    ├── # Plotly controls whose scene names no dragmode run Plotly's turntable default, so they are rejected the same way.
    ├── impls build Plotly controls whose scene carries no dragmode
    ├── with pytest.raises on the restricted-camera-pose-controls message
    │   └── calls assert_dash_no_camera_pose_clamps
    └── return
```

### Frontend

```text
frontend roll-lock expectations  # agent-conducted manual procedures, each conducted on both stacks by driving a running spatial display in a real browser and reading the camera the page reports
├── A display built with no lock axis left-drag-rotates as a free trackball, the horizon tilting freely as the drag continues.
├── A display built with a lock axis holds its horizon level at every pointer move of the same left-drag, the camera right axis perpendicular to the supplied axis.
├── A display that builds its own trackball controls hands the lock axis it is built with to those controls unchanged: create_dash_points_display and create_dash_mesh_display on Dash, renderPointsDisplay, renderMeshDisplay, renderSceneGraphDisplay, renderAabb3dDisplay, and renderLayeredDisplay on TS.
├── A roll-locked Dash display built inside a Dash callback after the page has loaded holds its lock exactly as one built into the initial layout does.
├── A roll-locked display answers a left-drag as yaw about the supplied axis plus pitch about the camera right axis, a pure-horizontal drag holding its polar angle off the axis and a pure-vertical drag holding its azimuth about it.
├── A roll-locked display's camera up coincides with the supplied axis only where its view direction is perpendicular to that axis.
├── A roll-locked display left-dragged straight up stops pitching at the pole on its supplied axis, the horizon level and the scene upright throughout, and a drag back down turns it off the pole again.
├── A roll-locked display renders no frame rolled off the lock through a long left-drag started inside its viewport, at any drag speed.
├── A roll-locked Dash display over data whose axis ranges differ by more than 4× holds a tilted lock axis upright on screen at every painted frame, from the first frame painted before the roll-lock callback first runs.
├── A Dash figure update that changes a roll-locked display's data extent paints no frame off the lock.
├── An aspectmode written onto a roll-locked Dash display, through a Plotly.relayout or a Dash figure update, paints no frame off the lock and leaves the next left-drag holding it.
├── A Dash callback whose Output is a roll-locked display's own id updates that display and leaves the page running.
├── A roll-locked display's left-drag turns its camera as far per pixel as its free counterpart's.
├── A camera or rotation target written directly onto a roll-locked display — a Plotly.relayout, a Dash figure update with or without a projection switch in it, a Dash figure update that drops the 3D trace and adds it back, a modebar reset-camera, a rotation-mode switch in either direction, or a projection switch on Dash; a controls.target or camera.position write on TS — renders no frame off the lock, and on Dash leaves the camera the layout stores on the lock.
├── A camera written onto a roll-locked display with its up on the far side of the axis keeps the eye it was written with, turned upright about its own view direction.
├── A roll-locked display right-drag-pans and wheel-zooms over the same range its free counterpart reaches.
├── Two camera-synced displays built with the same lockRoll setting end every left-drag on either one in the same pose, both roll-locked throughout.
├── A roll-locked TS display whose camera is written with its eye on its target keeps the pose it last held, renders no frame off the lock, and answers the next drag and wheel normally.
├── A roll-locked Dash display whose camera is written with its eye on its center keeps the pose it last held, renders no frame off the lock, and answers the next drag and wheel normally.
└── A burst of camera writes issued without awaiting on a roll-locked Dash display leaves the camera the layout stores on the lock.
```
