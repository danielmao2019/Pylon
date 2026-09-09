"""Tests for the Dash trackball camera controls and their guards."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import pytest
from dash import Dash, dcc, html

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCK_CALLBACK_SCRIPT_PATH,
    assert_dash_no_camera_pose_clamps,
    assert_dash_roll_lock,
    assert_dash_trackball_camera_controls,
    create_dash_trackball_camera_controls,
    register_dash_roll_lock_callback,
)
from data.viewer.utils.displays.mesh.dash.core_mesh_display import (
    TEXTURED_MESH_VIEWER_SCRIPT_PATH,
)

# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# The same direction at a length far from 1, so a construction that forwards the
# caller's axis unnormalized cannot pass.
NON_UNIT_LOCK_ROLL = (3.0, 9.0, -2.0)
# The component id the roll-locked graph is registered under in the callback tests.
ROLL_LOCKED_GRAPH_ID = "roll-locked-graph"
# Radians of pitch each simulated drag turns the panel's camera through, negative so the
# drag climbs toward the pole the camera starts nearest.
ROLL_LOCKED_PITCH_PER_DRAG_RADIANS = -0.2
# Simulated drags, enough that the accumulated pitch overshoots the pole several times
# over rather than merely reaching it.
ROLL_LOCKED_PITCH_DRAG_COUNT = 20
# One record for the panel's initial render, which is when Dash first fires the callback
# and so the only moment a start the drags cannot reach is handed to it, plus one record
# per simulated drag.
ROLL_LOCKED_PITCH_RECORD_COUNT = 1 + ROLL_LOCKED_PITCH_DRAG_COUNT
# Radians from the lock axis at or below which the camera counts as having reached the
# pole, generous next to the correction's own stopping distance.
ROLL_LOCKED_POLE_REACHED_RADIANS = 1e-3
# Magnitude of `right . axis` at or below which the camera right axis counts as
# perpendicular to the lock axis.
ROLL_LOCKED_PERPENDICULAR_TOLERANCE = 1e-9
# Deviation from 1 at or below which a basis vector the camera carries counts as unit
# length. A basis that is finite but not unit length is a camera whose view matrix
# carries a scale, which renders the scene at the wrong size rather than at no size.
ROLL_LOCKED_UNIT_LENGTH_TOLERANCE = 1e-9
# The camera eye the simulated panel comes up on where the start is not degenerate, off
# the lock axis so the pitch reaches the pole from a frame that is not already on it.
ROLL_LOCKED_OFF_AXIS_EYE = (1.25, 1.25, 1.25)
# Distance from the rotation target the degenerate starts place the eye at, matching the
# off-axis start's own radius so every start turns through the same sphere.
ROLL_LOCKED_START_RADIUS = 1.25 * math.sqrt(3.0)


def build_free_trackball_renderer_controls() -> str:
    """Build renderer-control source whose left-drag rotation leaves camera roll free.

    Args:
        None.

    Returns:
        JavaScript source carrying the trackball mouse mapping and a left-drag
        rotation that lets the camera right axis tilt with the drag.
    """
    return """
    domElement.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });
    domElement.addEventListener("mousedown", (event) => {
      pointerState.mode = event.button === 2 ? "pan" : "rotate";
    });
    domElement.addEventListener("wheel", (event) => {
      event.preventDefault();
    });
    camera.rotation.y -= dx * 0.005;
    camera.rotation.x -= dy * 0.005;
    """


def build_roll_locked_renderer_controls() -> str:
    """Build renderer-control source whose left-drag rotation holds the camera right axis.

    Args:
        None.

    Returns:
        JavaScript source carrying the trackball mouse mapping plus the roll-lock
        wiring that re-derives the camera right axis perpendicular to the supplied
        axis on every drag step and clamps the pitch so the camera up vector stays
        on that axis's own side.
    """
    return build_free_trackball_renderer_controls() + """
    container.dataset.cameraRollLock = JSON.stringify(rollLockAxis);
    container.dataset.cameraRightAxisConstraint = "perpendicular-to-roll-lock-axis";
    container.dataset.cameraUpAxisConstraint = "same-side-as-roll-lock-axis";
    cameraRightAxis.crossVectors(viewDirection, rollLockAxis).normalize();
    camera.up.crossVectors(cameraRightAxis, viewDirection).normalize();
    pitchAngle = Math.min(Math.max(pitchAngle, -polarAngle), Math.PI - polarAngle);
    """


def expected_camera_up(lock_roll: Tuple[float, float, float]) -> Dict[str, float]:
    """Compute the unit-length Plotly camera up vector a lock axis must produce.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.

    Returns:
        Dict with `x`, `y`, and `z` unit-length components.
    """
    length = math.sqrt(sum(component * component for component in lock_roll))
    return {
        "x": lock_roll[0] / length,
        "y": lock_roll[1] / length,
        "z": lock_roll[2] / length,
    }


# ================================================================================
# The Plotly gl3d camera configuration the Dash displays render
# ================================================================================


def test_no_axis_selects_the_free_roll_dragmode() -> None:
    """A caller that names no lock_roll gets the dragmode under which roll is reachable, identical to an explicit lock_roll=None construction."""
    defaulted_controls = create_dash_trackball_camera_controls()
    explicit_controls = create_dash_trackball_camera_controls(lock_roll=None)

    assert defaulted_controls == explicit_controls, (
        "Naming no lock_roll must carry the same rotation wiring as an explicit "
        "lock_roll=None construction. "
        f"{defaulted_controls=} {explicit_controls=}"
    )
    assert defaulted_controls["dragmode"] == "orbit", (
        "Naming no roll-lock axis must select the Plotly gl3d dragmode that leaves "
        "the camera up vector free to tilt, which is what an unlocked display means. "
        f"{defaulted_controls=}"
    )
    assert_dash_no_camera_pose_clamps(controls=defaulted_controls, lock_roll=None)


def test_no_axis_never_selects_the_pose_clamping_dragmode() -> None:
    """The unlocked construction never leaves the pose-clamping dragmode in force, which is what an omitted dragmode would run and what would roll-lock the display to world +Z."""
    controls = create_dash_trackball_camera_controls(lock_roll=None)

    assert "dragmode" in controls, (
        "A scene configuration naming no dragmode runs Plotly's own gl3d default, "
        "which is the turntable that pins the camera up vector to world +Z, so an "
        f"unlocked display must name its dragmode outright. {controls=}"
    )
    assert controls["dragmode"] != "turntable", (
        "plotly.js pins the camera up vector to (0, 0, 1) under the turntable "
        "dragmode, making roll unreachable, so an unlocked display that renders "
        f"turntable is roll-locked about an axis its caller never named. {controls=}"
    )


def test_a_supplied_axis_is_carried_into_the_rendered_camera() -> None:
    """Constructing with a lock_roll seeds the Plotly camera up vector with that supplied axis."""
    controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    assert controls["dragmode"] == "orbit", (
        "Roll-locked controls must select the Plotly gl3d dragmode that carries an "
        f"arbitrary camera up vector through re-render. {controls=}"
    )
    assert controls["camera"]["up"] == expected_camera_up(NON_AXIS_ALIGNED_LOCK_ROLL), (
        "Roll-locked controls must seed the camera up vector with the caller's axis. "
        f"{controls=} {NON_AXIS_ALIGNED_LOCK_ROLL=}"
    )


def test_the_roll_locked_dragmode_never_clamps_the_camera_up_vector() -> None:
    """The roll-locked branch never emits the pose-clamping dragmode, under which plotly.js discards the caller's axis outright."""
    controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    assert controls["dragmode"] != "turntable", (
        "plotly.js discards any camera up vector whose normalized z falls below "
        "0.999 under the turntable dragmode and substitutes (0, 0, 1), so a "
        "turntable-emitting roll lock renders the unlocked camera for every axis "
        f"more than ~2.5 degrees off world +Z. {controls=}"
    )
    assert_dash_no_camera_pose_clamps(
        controls=controls,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )


def test_a_non_unit_axis_is_normalized() -> None:
    """The caller's axis need not be unit length, so the same direction at any length pins the same camera up vector."""
    unit_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )
    scaled_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_UNIT_LOCK_ROLL,
    )

    assert unit_controls == scaled_controls, (
        "A roll-lock axis is a direction, so scaling it must not change the pinned "
        f"camera up vector. {unit_controls=} {scaled_controls=}"
    )
    up = scaled_controls["camera"]["up"]
    assert math.isclose(
        math.sqrt(up["x"] ** 2 + up["y"] ** 2 + up["z"] ** 2), 1.0, rel_tol=1e-9
    ), ("The pinned camera up vector must be unit length. " f"{up=}")


def test_roll_locked_controls_keep_every_other_degree_of_freedom_free() -> None:
    """Roll lock constrains roll alone, so a lock_roll construction still passes the mouse-mapping, no-orbit, and no-pose-clamp contracts."""
    controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    assert_dash_trackball_camera_controls(
        controls=controls,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )
    assert "center" not in controls["camera"], (
        "A roll-locked construction must leave the rotation target unpinned. "
        f"{controls=}"
    )


def test_assert_dash_roll_lock_rejects_an_ignored_flag() -> None:
    """A configuration that pins no camera up vector is rejected against a supplied axis, so the flag cannot be silently dropped."""
    controls = create_dash_trackball_camera_controls(lock_roll=None)

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(
            controls=controls,
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        )


def test_assert_dash_roll_lock_rejects_a_mismatched_axis() -> None:
    """A configuration pinned to a different axis than the caller supplied is rejected, so the caller's axis cannot be swapped for another."""
    controls = create_dash_trackball_camera_controls(lock_roll=(0.0, 0.0, 1.0))

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(
            controls=controls,
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        )


def test_assert_dash_roll_lock_rejects_an_unrequested_lock() -> None:
    """A lock_roll=None configuration that nonetheless pins a camera up vector is rejected, so the default construction cannot quietly become roll-locked."""
    controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    with pytest.raises(
        AssertionError,
        match="free trackball camera controls must leave camera roll unconstrained",
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=None)


def test_assert_dash_no_camera_pose_clamps_rejects_a_roll_restricting_dragmode() -> (
    None
):
    """The roll-pinning Plotly dragmode restricts rotation, so it is rejected when no axis is supplied."""
    with pytest.raises(
        AssertionError,
        match="restricted camera pose controls are forbidden",
    ):
        assert_dash_no_camera_pose_clamps(
            controls={"dragmode": "turntable"},
            lock_roll=None,
        )


def test_assert_dash_no_camera_pose_clamps_rejects_an_omitted_dragmode() -> None:
    """A configuration naming no dragmode runs Plotly's own gl3d turntable default, so it is rejected exactly as an explicit turntable is."""
    with pytest.raises(
        AssertionError,
        match="restricted camera pose controls are forbidden",
    ):
        assert_dash_no_camera_pose_clamps(controls={}, lock_roll=None)


# ================================================================================
# The clientside callback that re-imposes the roll lock after every drag
# ================================================================================


def build_roll_locked_app() -> Dash:
    """Build a Dash app holding one roll-locked graph and nothing else.

    Args:
        None.

    Returns:
        A Dash app whose layout is a single `dcc.Graph` with id `ROLL_LOCKED_GRAPH_ID`
        and no callbacks registered.
    """
    app = Dash(__name__)
    app.layout = html.Div(
        children=[
            dcc.Graph(
                id=ROLL_LOCKED_GRAPH_ID,
                figure=go.Figure(
                    layout=go.Layout(
                        scene=create_dash_trackball_camera_controls(
                            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
                        ),
                    ),
                ),
            ),
        ],
    )
    return app


def test_the_roll_lock_callback_is_registered_against_the_named_graph() -> None:
    """Registering the roll lock adds one clientside callback driven by the named graph's own camera changes."""
    app = build_roll_locked_app()

    register_dash_roll_lock_callback(
        app=app,
        graph_id=ROLL_LOCKED_GRAPH_ID,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    registrations = [
        registration
        for registration in app._callback_list
        if registration["clientside_function"] is not None
    ]
    assert len(registrations) == 1, (
        "Registering the roll lock must add exactly one clientside callback. "
        f"{registrations=}"
    )
    assert registrations[0]["inputs"] == [
        {"id": ROLL_LOCKED_GRAPH_ID, "property": "relayoutData"}
    ], (
        "The roll-lock callback must be driven by the named graph's own relayout "
        f"events, which is how every camera change reaches it. {registrations[0]=} "
        f"{ROLL_LOCKED_GRAPH_ID=}"
    )


def test_the_roll_lock_callback_carries_its_source_inline() -> None:
    """The callback source reaches the browser inlined in the app, so a consuming app needs no assets folder to serve it."""
    app = build_roll_locked_app()

    register_dash_roll_lock_callback(
        app=app,
        graph_id=ROLL_LOCKED_GRAPH_ID,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    inline_source = "\n".join(app._inline_scripts)
    assert "Plotly.relayout" in inline_source, (
        "The inlined callback must write the re-derived camera up vector back to the "
        f"panel. {inline_source=}"
    )
    assert ".js-plotly-plot" in inline_source, (
        "`dcc.Graph` renders its component id onto a wrapper div, so the inlined "
        "callback must resolve the Plotly graph div inside that wrapper. "
        f"{inline_source=}"
    )
    assert json.dumps(ROLL_LOCKED_GRAPH_ID) in inline_source, (
        "The inlined callback must carry the graph id it was registered against. "
        f"{inline_source=} {ROLL_LOCKED_GRAPH_ID=}"
    )


def test_the_roll_lock_callback_normalizes_the_caller_axis() -> None:
    """The caller's axis need not be unit length, so the same direction at any length inlines the same axis."""
    unit_app = build_roll_locked_app()
    scaled_app = build_roll_locked_app()

    register_dash_roll_lock_callback(
        app=unit_app,
        graph_id=ROLL_LOCKED_GRAPH_ID,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )
    register_dash_roll_lock_callback(
        app=scaled_app,
        graph_id=ROLL_LOCKED_GRAPH_ID,
        lock_roll=NON_UNIT_LOCK_ROLL,
    )

    assert unit_app._inline_scripts == scaled_app._inline_scripts, (
        "A roll-lock axis is a direction, so scaling it must not change the inlined "
        f"callback. {unit_app._inline_scripts=} {scaled_app._inline_scripts=}"
    )
    up = expected_camera_up(NON_UNIT_LOCK_ROLL)
    assert json.dumps([up["x"], up["y"], up["z"]]) in "\n".join(
        scaled_app._inline_scripts
    ), (
        "The inlined callback must carry the caller's axis at unit length. "
        f"{scaled_app._inline_scripts=} {up=}"
    )


def test_the_roll_lock_callback_rejects_a_zero_axis() -> None:
    """A zero axis names no direction to hold roll about, so registration refuses it rather than registering a callback that cannot normalize it."""
    app = build_roll_locked_app()

    with pytest.raises(
        AssertionError,
        match="Roll lock axis must be a non-zero 3-tuple of floats",
    ):
        register_dash_roll_lock_callback(
            app=app,
            graph_id=ROLL_LOCKED_GRAPH_ID,
            lock_roll=(0.0, 0.0, 0.0),
        )


def normalize_vector(
    vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Scale a non-zero world-space vector to unit length.

    Args:
        vector: Non-zero `(x, y, z)` world-space vector of any length.

    Returns:
        Tuple of the `(x, y, z)` components at unit length.
    """
    length = math.sqrt(sum(component * component for component in vector))
    assert length > 0, f"Cannot normalize a zero-length vector. {vector=}"
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def cross_vectors(
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Compute the cross product of two world-space vectors.

    Args:
        left: The `(x, y, z)` world-space vector on the left of the product.
        right: The `(x, y, z)` world-space vector on the right of the product.

    Returns:
        Tuple of the product's `(x, y, z)` components.
    """
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def build_eye_along_lock_axis(
    lock_roll: Tuple[float, float, float],
    radius: float,
) -> Tuple[float, float, float]:
    """Build the camera eye sitting on the lock axis itself, where the roll lock's own polar angle is degenerate.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.
        radius: Signed distance from the rotation target along the normalized axis.
            Positive puts the eye on the pole the axis points at, so the polar angle is
            0; negative puts it past the far pole, so the polar angle is pi.

    Returns:
        Tuple of the eye's `(x, y, z)` world-space coordinates.
    """
    axis = normalize_vector(lock_roll)
    return (axis[0] * radius, axis[1] * radius, axis[2] * radius)


def build_up_across_lock_axis(
    lock_roll: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Build a camera up vector perpendicular to the lock axis, which is what keeps a camera framed straight down that axis drawable.

    The view direction of a camera whose eye sits on the lock axis runs along that axis,
    so an up vector along it too would leave the camera with no screen-right axis at all
    and describe a pose no renderer can draw, whatever the roll lock does. Standing the
    up vector across the axis leaves the pose drawable while keeping the polar angle
    degenerate, which is the degeneracy the roll lock itself owns.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.

    Returns:
        Tuple of the up vector's unit-length `(x, y, z)` components.
    """
    axis = normalize_vector(lock_roll)
    least_aligned_basis = min(
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        key=lambda basis: abs(
            sum(
                axis_component * basis_component
                for axis_component, basis_component in zip(axis, basis, strict=True)
            ),
        ),
    )
    return normalize_vector(cross_vectors(axis, least_aligned_basis))


def build_inverted_up(
    lock_roll: Tuple[float, float, float],
    eye: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Build the camera up vector a panel reports once a drag has carried the view through a pole, which is the roll-locked up vector hanging on the axis's far side.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.
        eye: The camera eye's `(x, y, z)` world-space coordinates, off the lock axis,
            looking at a rotation target on the world origin.

    Returns:
        Tuple of the up vector's unit-length `(x, y, z)` components, perpendicular to
        the view direction as every up vector a gl3d panel reports is.
    """
    axis = normalize_vector(lock_roll)
    forward = normalize_vector((-eye[0], -eye[1], -eye[2]))
    right = normalize_vector(cross_vectors(forward, axis))
    up = normalize_vector(cross_vectors(right, forward))
    return (-up[0], -up[1], -up[2])


def build_roll_lock_pitch_harness_script(
    lock_roll: Tuple[float, float, float],
    eye: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> str:
    """Build the Node harness that drives the roll-lock callback through a pole-crossing pitch.

    The harness stands in for the panel: it holds the gl3d camera the callback reads,
    fires the callback once on the start pose the way Dash fires it on initial render,
    then turns the camera the way a Plotly `orbit` left-drag does (the eye and the camera
    up vector both rotating about the camera's own screen-right axis), hands the turned
    camera to the callback, and applies whatever `Plotly.relayout` the callback issues.

    Each callback call is watched for whether it can describe the camera it was handed
    at all, since a panel cannot survive a callback that aborts on the pose it reports
    and would abort again on every relayout after it, nor one that fails its own
    invariant and derives the pose it writes from the NaN that follows.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.
        eye: The `(x, y, z)` world-space coordinates the simulated panel's camera starts
            at, looking at a rotation target on the world origin.
        up: The `(x, y, z)` world-space up vector the simulated panel's camera starts
            with, non-parallel to the view direction so the start pose is one a renderer
            can draw.

    Returns:
        JavaScript source that prints one JSON record for the initial render and one per
        simulated drag, each carrying `drag`, `callback_error`,
        `callback_assertion_failures`, `up_along_axis`, `right_along_axis`, `polar`,
        `written_eye`, and `written_up` measured on the camera the callback left behind.
    """
    return """
const ROLL_LOCK_SOURCE = %s;
const GRAPH_ID = %s;
const LOCK_ROLL = %s;
const EYE = %s;
const UP = %s;
const PITCH_PER_DRAG = %s;
const DRAG_COUNT = %s;

function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function normalize(a) { const l = Math.sqrt(dot(a, a)); return [a[0] / l, a[1] / l, a[2] / l]; }
function toRecord(a) { return { x: a[0], y: a[1], z: a[2] }; }
function toVector(r) { return [r.x, r.y, r.z]; }
function rotate(v, axis, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
}

const axis = normalize(LOCK_ROLL);
let camera = { eye: toRecord(EYE), center: toRecord([0, 0, 0]), up: toRecord(UP) };

function pitchDrag(angle) {
  const eye = toVector(camera.eye);
  const center = toVector(camera.center);
  const up = toVector(camera.up);
  const forward = normalize(subtract(center, eye));
  const right = normalize(cross(forward, up));
  camera = {
    eye: toRecord(add(center, rotate(subtract(eye, center), right, angle))),
    center: camera.center,
    up: toRecord(normalize(rotate(up, right, angle))),
  };
}

const graphDiv = { _fullLayout: { scene: { _scene: { getCamera: () => camera } } } };
globalThis.document = {
  getElementById: (id) =>
    id === GRAPH_ID ? { querySelector: (s) => (s === ".js-plotly-plot" ? graphDiv : null) } : null,
};
globalThis.window = { dash_clientside: { no_update: null } };
globalThis.Plotly = {
  relayout: (div, update) => {
    if (update["scene.camera.eye"] !== undefined) {
      camera = { eye: update["scene.camera.eye"], center: camera.center, up: camera.up };
    }
    if (update["scene.camera.up"] !== undefined) {
      camera = { eye: camera.eye, center: camera.center, up: update["scene.camera.up"] };
    }
    return { then: (settle) => { settle(); } };
  },
};

// The callback reports a state it cannot describe in one of two ways: it throws, or it
// fails one of its own `console.assert` invariants and carries the NaN onward. A harness
// that watches only the throw is blind to the second, which is how a pose built out of a
// zero-length normalization reaches the panel with nothing raised. Both are captured
// against the step that provoked them.
let stepAssertionFailures = [];
console.assert = function (condition) {
  if (condition) {
    return;
  }
  stepAssertionFailures.push(
    Array.prototype.slice.call(arguments, 1).map(function (value) { return String(value); }).join(" "),
  );
};

const callback = eval(ROLL_LOCK_SOURCE)(GRAPH_ID, axis);
const records = [];
// Drag 0 is the panel's initial render, which Dash fires the callback on before any
// drag has moved the camera, so it is the only step that reaches the callback with the
// start pose the caller framed the panel with.
for (let drag = 0; drag <= DRAG_COUNT; drag += 1) {
  if (drag > 0) {
    pitchDrag(PITCH_PER_DRAG);
  }
  let callbackError = null;
  stepAssertionFailures = [];
  try {
    callback(null);
  } catch (error) {
    callbackError = String(error && error.message !== undefined ? error.message : error);
  }
  const offset = subtract(toVector(camera.eye), toVector(camera.center));
  const forward = normalize(scale(offset, -1));
  const right = normalize(cross(forward, toVector(camera.up)));
  // The camera's own basis is recorded alongside the two dot products because both of
  // those read a direction and neither reads a length: a pose whose up vector is short,
  // long, or non-finite is a camera the panel renders the scene at the wrong size
  // through, and both dot products pass it unremarked.
  records.push({
    drag: drag,
    callback_error: callbackError,
    callback_assertion_failures: stepAssertionFailures,
    up_along_axis: dot(normalize(toVector(camera.up)), axis),
    right_along_axis: dot(right, axis),
    polar: Math.acos(Math.max(-1, Math.min(1, dot(normalize(offset), axis)))),
    written_eye: toVector(camera.eye),
    written_up: toVector(camera.up),
  });
}
process.stdout.write(JSON.stringify(records));
""" % (
        json.dumps(ROLL_LOCK_CALLBACK_SCRIPT_PATH.read_text()),
        json.dumps(ROLL_LOCKED_GRAPH_ID),
        json.dumps(list(lock_roll)),
        json.dumps(list(eye)),
        json.dumps(list(up)),
        json.dumps(ROLL_LOCKED_PITCH_PER_DRAG_RADIANS),
        json.dumps(ROLL_LOCKED_PITCH_DRAG_COUNT),
    )


def run_roll_lock_pitch_harness(harness_path: Path) -> List[Dict[str, Any]]:
    """Run the roll-lock pitch harness under Node and read back its per-step records.

    Args:
        harness_path: Path the harness source was written to.

    Returns:
        One dict for the initial render and one per simulated drag, each carrying `drag`
        as an int, `callback_error` as the message of whatever the callback raised on
        that step or None, `callback_assertion_failures` as the messages of whatever
        internal invariants failed on that step, `up_along_axis`, `right_along_axis`,
        and `polar` as floats, and `written_eye` and `written_up` as component lists
        with None standing for any non-finite number JSON cannot express.
    """
    completed_process = subprocess.run(
        args=["node", str(harness_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, (
        "Expected the roll-lock pitch Node harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    return json.loads(completed_process.stdout)


def is_unit_length(components: List[Optional[float]]) -> bool:
    """Report whether a basis the camera carries is a finite, unit-length vector.

    Args:
        components: The vector's components as the harness's JSON carried them, with
            None standing for any non-finite number JSON cannot express.

    Returns:
        True when every component is a finite number and the vector's length is 1 to
        within `ROLL_LOCKED_UNIT_LENGTH_TOLERANCE`, False otherwise.
    """
    if any(component is None for component in components):
        return False
    return math.isclose(
        math.sqrt(sum(component * component for component in components)),
        1.0,
        abs_tol=ROLL_LOCKED_UNIT_LENGTH_TOLERANCE,
    )


def assert_roll_locked_camera(records: List[Dict[str, Any]]) -> None:
    """Assert every step left the callback standing, the horizon level, and the camera a real camera.

    The three clauses are what a person looking at the panel would call wrong. A callback
    that cannot describe the camera it was handed leaves the panel with no roll lock at
    all from that relayout on, which the two measurements below cannot see because both
    are taken on the pose that call never wrote; a camera right axis off the lock axis
    tips the horizon; and a camera whose basis is not unit length is not a camera at all,
    which the right-axis measurement also cannot see because it reads a direction and
    never a length. Soundness is asserted first for that reason: the other two describe a
    pose the callback stood behind, and a callback that aborted or carried a NaN through
    its own failed invariant stood behind nothing.

    Args:
        records: One dict per harness step, each carrying `drag`, `callback_error`,
            `callback_assertion_failures`, `right_along_axis`, `written_eye`, and
            `written_up` as `run_roll_lock_pitch_harness` returns them.

    Returns:
        None.
    """
    unsound_records = [
        record
        for record in records
        if record["callback_error"] is not None or record["callback_assertion_failures"]
    ]
    assert not unsound_records, (
        "The roll lock must hold for every camera the panel can report, so the callback "
        "must describe each of them rather than reaching a state it cannot: aborting "
        "leaves the panel unlocked from that relayout on, and a failed internal "
        "invariant leaves it deriving the pose from a NaN that only the next write "
        f"decides whether to show. {unsound_records=} {records=}"
    )
    tilted_records = [
        record
        for record in records
        if abs(record["right_along_axis"]) > ROLL_LOCKED_PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "A roll-locked camera must keep its right axis perpendicular to the lock axis, "
        f"which is what holds the horizon level. {tilted_records=} {records=}"
    )
    unreal_records = [
        record
        for record in records
        if any(component is None for component in record["written_eye"])
        or not is_unit_length(record["written_up"])
    ]
    assert not unreal_records, (
        "A roll-locked camera must stay a camera a renderer can draw: a finite eye, and "
        "an up vector that is unit length, since a short or long up vector scales the "
        "camera's view matrix and renders the scene at the wrong size. "
        f"{unreal_records=} {records=}"
    )


def run_roll_lock_pitch_drags(
    tmp_path: Path,
    eye: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> List[Dict[str, Any]]:
    """Drive the roll-lock callback through the pitch drags from one start pose and read back its records.

    Args:
        tmp_path: Directory the harness source is written to.
        eye: The `(x, y, z)` world-space coordinates the simulated panel's camera starts
            at, looking at a rotation target on the world origin.
        up: The `(x, y, z)` world-space up vector the simulated panel's camera starts
            with.

    Returns:
        One dict for the initial render and one per simulated drag, as
        `run_roll_lock_pitch_harness` returns them.
    """
    harness_path = tmp_path / "roll_lock_pitch_harness.js"
    harness_path.write_text(
        build_roll_lock_pitch_harness_script(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            eye=eye,
            up=up,
        ),
    )

    records = run_roll_lock_pitch_harness(harness_path=harness_path)

    assert len(records) == ROLL_LOCKED_PITCH_RECORD_COUNT, (
        "The harness must report one record for the initial render and one per "
        f"simulated drag. {len(records)=} {ROLL_LOCKED_PITCH_RECORD_COUNT=}"
    )
    return records


def test_the_roll_lock_callback_stops_a_pitch_at_the_pole(tmp_path: Path) -> None:
    """A pitch that reaches the pole stops there rather than carrying the view through it, so the camera never comes out the far side."""
    records = run_roll_lock_pitch_drags(
        tmp_path=tmp_path,
        eye=ROLL_LOCKED_OFF_AXIS_EYE,
        up=normalize_vector(NON_AXIS_ALIGNED_LOCK_ROLL),
    )

    polar_angles = [record["polar"] for record in records]
    assert min(polar_angles) <= ROLL_LOCKED_POLE_REACHED_RADIANS, (
        "The simulated pitch must actually reach the pole, or nothing about the pole "
        f"is under test. {polar_angles=} {ROLL_LOCKED_POLE_REACHED_RADIANS=}"
    )
    assert polar_angles[-1] <= ROLL_LOCKED_POLE_REACHED_RADIANS, (
        "Every drag past the pole must leave the camera at the pole rather than "
        f"carrying it out the far side. {polar_angles=}"
    )
    assert_roll_locked_camera(records=records)


def test_the_roll_lock_callback_never_lets_a_pitch_invert_the_camera(
    tmp_path: Path,
) -> None:
    """A pole-reaching pitch leaves the camera up vector on the lock axis's own side at every drag, which a camera right axis perpendicular to that axis never says on its own."""
    records = run_roll_lock_pitch_drags(
        tmp_path=tmp_path,
        eye=ROLL_LOCKED_OFF_AXIS_EYE,
        up=normalize_vector(NON_AXIS_ALIGNED_LOCK_ROLL),
    )

    assert_roll_locked_camera(records=records)

    minimum_up_along_axis = min(record["up_along_axis"] for record in records)
    assert minimum_up_along_axis >= 0, (
        "A pitch the roll lock carries to the pole must leave the camera up vector on "
        "the lock axis's own side at every drag, so the smallest `up . axis` the drags "
        f"reach must never go negative. {minimum_up_along_axis=} {records=}"
    )


def test_the_roll_lock_callback_holds_from_an_eye_on_the_lock_axis(
    tmp_path: Path,
) -> None:
    """A panel framed straight down the lock axis reports a camera whose polar angle is 0, and the roll lock must hold from there rather than aborting on the axis it is asked to hold about."""
    records = run_roll_lock_pitch_drags(
        tmp_path=tmp_path,
        eye=build_eye_along_lock_axis(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            radius=ROLL_LOCKED_START_RADIUS,
        ),
        up=build_up_across_lock_axis(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
    )

    assert_roll_locked_camera(records=records)


def test_the_roll_lock_callback_holds_from_an_eye_past_the_far_pole(
    tmp_path: Path,
) -> None:
    """A panel framed straight up the lock axis reports a camera whose polar angle is pi, the other end of the same degeneracy, and the roll lock must hold from there too."""
    records = run_roll_lock_pitch_drags(
        tmp_path=tmp_path,
        eye=build_eye_along_lock_axis(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            radius=-ROLL_LOCKED_START_RADIUS,
        ),
        up=build_up_across_lock_axis(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
    )

    assert_roll_locked_camera(records=records)


def test_the_roll_lock_callback_holds_from_an_already_inverted_camera(
    tmp_path: Path,
) -> None:
    """A panel that comes up already hanging upside down is a start the drags cannot reach, and the roll lock must put it back on the axis's own side from the first render."""
    records = run_roll_lock_pitch_drags(
        tmp_path=tmp_path,
        eye=ROLL_LOCKED_OFF_AXIS_EYE,
        up=build_inverted_up(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            eye=ROLL_LOCKED_OFF_AXIS_EYE,
        ),
    )

    assert_roll_locked_camera(records=records)

    minimum_up_along_axis = min(record["up_along_axis"] for record in records)
    assert minimum_up_along_axis >= 0, (
        "A panel that comes up hanging upside down must be put back on the lock axis's "
        "own side from the first render, so the smallest `up . axis` across the steps "
        f"must never go negative. {minimum_up_along_axis=} {records=}"
    )


# A camera whose eye sits on its own rotation target is the one remaining degenerate pose
# the callback's own normalization cannot describe, and it has no test because a Plotly
# gl3d panel cannot report it. `Scene.initializeGLCamera` builds the panel's camera with
# `zoomMin: 0.01, zoomMax: 100`, which become the view controller's radius bounds
# `[log(0.01), log(100)]`; the eye-to-target distance is stored as that bounded radius
# and `setDistance` additionally ignores any non-positive distance outright, so no drag,
# no wheel, and no layout-seeded camera reaches a distance of 0. Measured against the
# plotly.js bundle the repo's `plotly` 6.7.0 ships.


# ================================================================================
# The three.js viewer's camera-control JavaScript source
# ================================================================================


def test_the_threejs_viewer_source_passes_the_trackball_contract() -> None:
    """The shipped three.js mesh viewer source satisfies every trackball contract, so the display's guard keeps guarding it."""
    controls = TEXTURED_MESH_VIEWER_SCRIPT_PATH.read_text()

    assert_dash_trackball_camera_controls(controls=controls, lock_roll=None)


def test_free_trackball_source_leaves_camera_roll_unconstrained() -> None:
    """Renderer source whose left-drag rotation carries the camera up vector passes the free-trackball contract and fails the roll-locked one."""
    controls = build_free_trackball_renderer_controls()

    assert_dash_roll_lock(controls=controls, lock_roll=None)
    with pytest.raises(AssertionError, match="perpendicular"):
        assert_dash_roll_lock(
            controls=controls,
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        )


def test_roll_locked_source_holds_the_camera_right_axis_and_up_vector() -> None:
    """Renderer source that re-derives the camera right axis and clamps the pitch passes the roll-locked contract and fails the free-trackball one."""
    controls = build_roll_locked_renderer_controls()

    assert_dash_roll_lock(controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
    with pytest.raises(
        AssertionError,
        match="free trackball camera controls must leave camera roll unconstrained",
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=None)
