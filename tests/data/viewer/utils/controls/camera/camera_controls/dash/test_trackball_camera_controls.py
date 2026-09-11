"""Tests for the Dash trackball camera controls and their roll-lock guards."""

import json
import math
from typing import Optional, Tuple

import plotly.graph_objects as go
import pytest
from dash import Dash, dcc

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCK_CALLBACK_SCRIPT_PATH,
    apply_dash_trackball_camera_controls,
    assert_dash_no_camera_pose_clamps,
    assert_dash_roll_lock,
    create_dash_trackball_camera_controls,
)
from data.viewer.utils.displays.mesh.dash.core_mesh_display import (
    TEXTURED_MESH_VIEWER_SCRIPT_PATH,
)

# Component id every test display carries.
GRAPH_ID = "trackball-graph"
# Deliberately non-axis-aligned, so nothing passes by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# The same direction at ten times the length, so an axis forwarded unnormalized cannot pass.
NON_UNIT_LOCK_ROLL = (3.0, 9.0, -2.0)
# Renderer source wiring the trackball mouse mapping, whose left-drag turns camera.up together with the eye, so roll moves freely with the drag.
FREE_TRACKBALL_RENDERER_SOURCE = """
canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});
canvas.addEventListener("mousedown", (event) => {
  dragMode = event.button === 2 ? "pan" : "rotate";
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  camera.position.multiplyScalar(1 + event.deltaY * 0.001);
});
canvas.addEventListener("mousemove", (event) => {
  if (dragMode !== "rotate") {
    return;
  }
  const dragRotation = new THREE.Quaternion().setFromAxisAngle(dragAxis, dragAngle);
  camera.position.applyQuaternion(dragRotation);
  camera.up.applyQuaternion(dragRotation);
  camera.lookAt(0, 0, 0);
});
"""


def test_no_axis_puts_the_display_under_the_free_roll_dragmode() -> None:
    """A display put under the controls with no lock_roll runs the orbit dragmode and pins no camera.up, identical to an explicit lock_roll=None application.

    Args:
        None.

    Returns:
        None.
    """
    default_display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )
    explicit_display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    apply_dash_trackball_camera_controls(app=Dash(__name__), display=default_display)
    apply_dash_trackball_camera_controls(
        app=Dash(__name__), display=explicit_display, lock_roll=None
    )

    default_scene = default_display.figure.layout.scene.to_plotly_json()
    explicit_scene = explicit_display.figure.layout.scene.to_plotly_json()
    assert default_scene == {"dragmode": "orbit"}, (
        "Expected the free trackball to run the orbit dragmode and pin no camera. "
        f"{default_scene=}"
    )
    assert default_scene == explicit_scene, (
        "Expected omitting lock_roll to match an explicit lock_roll=None. "
        f"{default_scene=} {explicit_scene=}"
    )


def test_no_axis_registers_no_roll_lock_callback() -> None:
    """A display put under the controls with no lock_roll gets no clientside callback, so nothing constrains its roll through a drag.

    Args:
        None.

    Returns:
        None.
    """
    app = Dash(__name__)
    display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    apply_dash_trackball_camera_controls(app=app, display=display)

    assert app.callback_map == {}, (
        "Expected no callback on a free-trackball display. " f"{app.callback_map=}"
    )


def test_a_supplied_axis_seeds_the_normalized_axis_as_camera_up() -> None:
    """A display put under the controls with a lock_roll carries that axis, normalized, as its figure's camera.up under the orbit dragmode.

    Args:
        None.

    Returns:
        None.
    """
    display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    apply_dash_trackball_camera_controls(
        app=Dash(__name__), display=display, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )

    length = math.sqrt(sum(component**2 for component in NON_AXIS_ALIGNED_LOCK_ROLL))
    expected_up = {
        "x": NON_AXIS_ALIGNED_LOCK_ROLL[0] / length,
        "y": NON_AXIS_ALIGNED_LOCK_ROLL[1] / length,
        "z": NON_AXIS_ALIGNED_LOCK_ROLL[2] / length,
    }
    scene = display.figure.layout.scene.to_plotly_json()
    assert scene["dragmode"] == "orbit", (
        "Expected the roll-locked display to run the orbit dragmode. " f"{scene=}"
    )
    assert scene["camera"]["up"] == pytest.approx(expected_up, abs=1e-12), (
        "Expected camera.up to be the normalized lock_roll. " f"{scene=} {expected_up=}"
    )


def test_a_non_unit_axis_is_normalized() -> None:
    """The caller's axis need not be unit length, so the same direction at any length pins the same camera up vector.

    Args:
        None.

    Returns:
        None.
    """
    unit_display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )
    scaled_display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    apply_dash_trackball_camera_controls(
        app=Dash(__name__), display=unit_display, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )
    apply_dash_trackball_camera_controls(
        app=Dash(__name__), display=scaled_display, lock_roll=NON_UNIT_LOCK_ROLL
    )

    unit_up = unit_display.figure.layout.scene.camera.up.to_plotly_json()
    scaled_up = scaled_display.figure.layout.scene.camera.up.to_plotly_json()
    assert scaled_up == pytest.approx(unit_up, abs=1e-12), (
        "Expected one direction at two lengths to pin one camera up vector. "
        f"{unit_up=} {scaled_up=}"
    )
    assert math.isclose(
        math.sqrt(sum(component**2 for component in scaled_up.values())), 1.0
    ), ("Expected the pinned camera up vector to be unit length. " f"{scaled_up=}")


def test_a_supplied_axis_registers_the_roll_lock_callback_on_the_display() -> None:
    """A display put under the controls with a lock_roll gets the roll-lock clientside callback on its own component id, so the lock holds through the drag rather than at its end.

    Args:
        None.

    Returns:
        None.
    """
    app = Dash(__name__)
    display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    apply_dash_trackball_camera_controls(
        app=app, display=display, lock_roll=NON_UNIT_LOCK_ROLL
    )

    callbacks = list(app.callback_map.values())
    assert len(callbacks) == 1, (
        "Expected exactly one callback on a roll-locked display. " f"{callbacks=}"
    )
    assert callbacks[0]["inputs"] == [{"id": GRAPH_ID, "property": "relayoutData"}], (
        "Expected the callback's one input to be the display's relayoutData. "
        f"{callbacks[0]['inputs']=}"
    )
    length = math.sqrt(sum(component**2 for component in NON_UNIT_LOCK_ROLL))
    axis = [component / length for component in NON_UNIT_LOCK_ROLL]
    expected_source = "(%s)(%s, %s)" % (
        ROLL_LOCK_CALLBACK_SCRIPT_PATH.read_text(),
        json.dumps(GRAPH_ID),
        json.dumps(axis),
    )
    assert len(app._inline_scripts) == 1, (
        "Expected exactly one inline clientside source. " f"{len(app._inline_scripts)=}"
    )
    assert expected_source in app._inline_scripts[0], (
        "Expected the inline source to be roll_lock.js called on the display id "
        "and the normalized lock_roll. "
        f"{GRAPH_ID=} {axis=} {app._inline_scripts[0][-200:]=}"
    )


def test_apply_rejects_a_display_without_an_id() -> None:
    """A display carrying no component id is rejected, since the roll-lock callback could address no graph.

    Args:
        None.

    Returns:
        None.
    """
    display = dcc.Graph(
        figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    with pytest.raises(AssertionError, match="Display must carry a component id"):
        apply_dash_trackball_camera_controls(
            app=Dash(__name__), display=display, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
        )


def test_apply_rejects_a_zero_axis() -> None:
    """A zero-length lock_roll names no direction, so it is rejected rather than normalized into a NaN camera.up.

    Args:
        None.

    Returns:
        None.
    """
    display = dcc.Graph(
        id=GRAPH_ID, figure=go.Figure(data=[go.Scatter3d(x=[0.0], y=[0.0], z=[0.0])])
    )

    with pytest.raises(AssertionError, match="non-zero 3-tuple of floats"):
        apply_dash_trackball_camera_controls(
            app=Dash(__name__), display=display, lock_roll=(0.0, 0.0, 0.0)
        )


def test_roll_locked_controls_keep_every_other_degree_of_freedom_free() -> None:
    """Roll lock constrains roll alone, so a supplied lock_roll configuration still passes the mouse-mapping, no-orbit, and no-pose-clamp contracts.

    Args:
        None.

    Returns:
        None.
    """
    length = math.sqrt(sum(component**2 for component in NON_AXIS_ALIGNED_LOCK_ROLL))
    controls = {
        "dragmode": "orbit",
        "camera": {
            "up": {
                "x": NON_AXIS_ALIGNED_LOCK_ROLL[0] / length,
                "y": NON_AXIS_ALIGNED_LOCK_ROLL[1] / length,
                "z": NON_AXIS_ALIGNED_LOCK_ROLL[2] / length,
            },
        },
    }

    constructed = create_dash_trackball_camera_controls(
        renderer_controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )

    assert constructed == controls, (
        "Expected the roll-locked configuration to pass every contract unchanged. "
        f"{constructed=} {controls=}"
    )


def test_the_threejs_viewer_source_passes_the_trackball_contract() -> None:
    """The shipped three.js mesh viewer source, handed over the way the mesh display hands it, satisfies every trackball contract and comes back unchanged.

    Args:
        None.

    Returns:
        None.
    """
    source = TEXTURED_MESH_VIEWER_SCRIPT_PATH.read_text(encoding="utf-8").replace(
        "__CAMERA_SYNC_SCRIPT__", ""
    )

    constructed = create_dash_trackball_camera_controls(renderer_controls=source)

    assert constructed is source, (
        "Expected the three.js viewer source to come back unchanged. "
        f"{len(constructed)=} {len(source)=}"
    )


def test_free_trackball_source_leaves_camera_roll_unconstrained() -> None:
    """Renderer source whose left-drag rotation carries the camera up vector passes the free-trackball contract and fails the roll-locked one.

    Args:
        None.

    Returns:
        None.
    """
    constructed = create_dash_trackball_camera_controls(
        renderer_controls=FREE_TRACKBALL_RENDERER_SOURCE
    )

    assert constructed is FREE_TRACKBALL_RENDERER_SOURCE, (
        "Expected the free-trackball source to come back unchanged. " f"{constructed=}"
    )
    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(
            controls=FREE_TRACKBALL_RENDERER_SOURCE,
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        )


def test_the_roll_lock_source_holds_the_camera_right_axis_and_up_vector() -> None:
    """The shipped roll_lock.js source passes the roll-locked contract and fails the free-trackball one.

    Args:
        None.

    Returns:
        None.
    """
    script = ROLL_LOCK_CALLBACK_SCRIPT_PATH.read_text()

    assert_dash_roll_lock(controls=script, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
    with pytest.raises(
        AssertionError,
        match="free trackball camera controls must leave camera roll unconstrained",
    ):
        assert_dash_roll_lock(controls=script)


def test_assert_dash_roll_lock_rejects_an_ignored_flag() -> None:
    """A supplied-lock_roll configuration that pins no camera.up is rejected, so the flag cannot be silently dropped.

    Args:
        None.

    Returns:
        None.
    """
    controls = {"dragmode": "orbit"}

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)


def test_assert_dash_roll_lock_rejects_a_mismatched_axis() -> None:
    """A configuration pinned to a different axis than the caller supplied is rejected, so the caller's axis cannot be swapped for another.

    Args:
        None.

    Returns:
        None.
    """
    controls = {"dragmode": "orbit", "camera": {"up": {"x": 0.0, "y": 0.0, "z": 1.0}}}

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)


def test_assert_dash_roll_lock_rejects_a_source_without_the_polar_band() -> None:
    """Roll-locked renderer source that holds the camera right axis perpendicular but bands no polar angle short of the poles is rejected, so a drag through a pole cannot hang the scene upside down.

    Args:
        None.

    Returns:
        None.
    """
    source = """
function rollLockDragStep(yaw, pitch) {
  const forward = center.clone().sub(camera.position).normalize();
  const right = new THREE.Vector3().crossVectors(forward, rollLockAxis).normalize();
  camera.position.sub(center).applyAxisAngle(rollLockAxis, yaw);
  camera.position.applyAxisAngle(right, pitch).add(center);
  camera.up.crossVectors(right, center.clone().sub(camera.position)).normalize();
}
"""

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera up vector on the "
            "supplied axis's side"
        ),
    ):
        assert_dash_roll_lock(controls=source, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)


def test_assert_dash_roll_lock_rejects_an_unrequested_lock() -> None:
    """A lock_roll=None configuration that nonetheless pins camera.up is rejected, so the default construction cannot quietly become roll-locked.

    Args:
        None.

    Returns:
        None.
    """
    controls = {"dragmode": "orbit", "camera": {"up": {"x": 0.0, "y": 0.0, "z": 1.0}}}

    with pytest.raises(
        AssertionError,
        match="free trackball camera controls must leave camera roll unconstrained",
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=None)


@pytest.mark.parametrize("lock_roll", [None, NON_AXIS_ALIGNED_LOCK_ROLL])
def test_assert_dash_no_camera_pose_clamps_rejects_the_pose_clamping_dragmode(
    lock_roll: Optional[Tuple[float, float, float]],
) -> None:
    """The turntable dragmode pins camera.up onto world +Z, so it is rejected as a pose clamp whether or not an axis is supplied.

    Args:
        lock_roll: Axis supplied alongside the configuration, or None for the free trackball.

    Returns:
        None.
    """
    controls = {"dragmode": "turntable"}

    with pytest.raises(AssertionError, match="restricted camera pose controls"):
        assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)


@pytest.mark.parametrize("lock_roll", [None, NON_AXIS_ALIGNED_LOCK_ROLL])
def test_assert_dash_no_camera_pose_clamps_rejects_an_omitted_dragmode(
    lock_roll: Optional[Tuple[float, float, float]],
) -> None:
    """A configuration naming no dragmode runs Plotly's turntable default, so it is rejected the same way.

    Args:
        lock_roll: Axis supplied alongside the configuration, or None for the free trackball.

    Returns:
        None.
    """
    controls = {"camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}}}

    with pytest.raises(AssertionError, match="restricted camera pose controls"):
        assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
