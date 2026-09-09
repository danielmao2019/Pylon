"""Tests for the TypeScript trackball camera controls' roll lock."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[9]
TRACKBALL_CAMERA_CONTROLS_SCRIPT_PATH = (
    REPO_ROOT
    / "data/viewer/utils/controls/camera/camera_controls/ts/frontend"
    / "trackball_camera_controls.ts"
)
# The repo's one Node package, whose `three`, `jsdom`, and `tsx` the harness runs the
# module's real rotation against rather than against a re-implementation of it.
WEB_NODE_MODULES_PATH = REPO_ROOT / "web/node_modules"
TSX_EXECUTABLE_PATH = WEB_NODE_MODULES_PATH / ".bin/tsx"
# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# Pixels of vertical pointer travel each simulated drag reports, which the controls turn
# into pitch at their own rotate speed; positive so the drag climbs toward the pole the
# default camera starts nearest.
ROLL_LOCKED_PITCH_PIXELS_PER_DRAG = 40
# Simulated drags, enough that the accumulated pitch overshoots the pole several times
# over rather than merely reaching it.
ROLL_LOCKED_PITCH_DRAG_COUNT = 20
# One record for the pose the controls come up on, which is the only moment a start the
# drags cannot reach is under test, plus one record per simulated drag.
ROLL_LOCKED_PITCH_RECORD_COUNT = 1 + ROLL_LOCKED_PITCH_DRAG_COUNT
# Radians from the lock axis at or below which the camera counts as having reached the
# pole, generous next to the controls' own stopping distance.
ROLL_LOCKED_POLE_REACHED_RADIANS = 1e-3
# Radians from the pole the camera must reach for a drag away from it to count as having
# moved the camera at all: far outside the band the controls hold it in, and well inside
# the pitch one drag asks for.
ROLL_LOCKED_POLE_DEPARTURE_RADIANS = 0.1
# Magnitude of `right . axis` at or below which the camera right axis counts as
# perpendicular to the lock axis.
ROLL_LOCKED_PERPENDICULAR_TOLERANCE = 1e-9
# Deviation from 1 at or below which a basis vector the camera carries counts as unit
# length. A basis that is finite but not unit length is a camera whose world matrix
# carries a scale, which renders the scene at the wrong size rather than at no size.
ROLL_LOCKED_UNIT_LENGTH_TOLERANCE = 1e-9
# The camera eye the simulated viewer comes up on where the start is not degenerate, off
# the lock axis so the pitch reaches the pole from a frame that is not already on it, and
# the up vector it comes up with.
ROLL_LOCKED_OFF_AXIS_EYE = (0.0, 0.0, 1.0)
ROLL_LOCKED_OFF_AXIS_UP = (0.0, 1.0, 0.0)
# The rotation target the simulated viewer looks at, which the degenerate starts measure
# their whole eye offset along the lock axis from.
ROLL_LOCKED_ROTATION_TARGET = (0.0, 0.0, 0.0)
# Distance from the rotation target the degenerate starts place the eye at, matching the
# off-axis start's own radius so every start turns through the same sphere.
ROLL_LOCKED_START_RADIUS = 1.0


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


def build_up_across_lock_axis(
    lock_roll: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Build a camera up vector perpendicular to the lock axis, which is what keeps a camera framed straight down that axis drawable.

    The view direction of a camera whose eye sits on the lock axis runs along that axis,
    so an up vector along it too would leave the camera with no screen-right axis at all
    and describe a pose no renderer can draw, whatever the roll lock does. Standing the
    up vector across the axis leaves the pose drawable while keeping the polar angle
    degenerate, which is the degeneracy the roll lock itself owns and which the controls
    derive their own right axis from the eye offset alone to reach.

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


def build_roll_lock_pitch_harness_script(
    lock_roll: Tuple[float, float, float],
    eye: Tuple[float, float, float],
    eye_offset_along_lock_axis: float,
    up: Tuple[float, float, float],
    pitch_pixels_per_drag: int,
) -> str:
    """Build the Node harness that drives the roll-locked controls through a pole-crossing pitch.

    The harness stands in for the browser: it mounts the real controls on a jsdom
    canvas over a real three camera, records the pose the controls come up on, holds a
    left-drag down, and walks the pointer along the canvas so every drag step reaches
    the controls' own rotation.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.
        eye: The `(x, y, z)` world-space coordinates the simulated viewer's camera
            starts at, before the lock-axis offset below is added, looking at a rotation
            target on the world origin.
        eye_offset_along_lock_axis: Signed distance the eye is then moved along the
            normalized lock axis, 0 to leave the coordinates above alone. A start on the
            lock axis is named this way rather than as coordinates because three
            normalizes by multiplying by the reciprocal of the length, so an axis
            normalized anywhere else lands an ulp off the one the controls hold, and the
            cross product that collapses on the axis comes out near zero rather than
            zero. Building the eye from the controls' own axis is what puts it exactly
            on the pole.
        up: The `(x, y, z)` world-space up vector the simulated viewer's camera starts
            with, non-parallel to the view direction so the start pose is one a renderer
            can draw.
        pitch_pixels_per_drag: Signed pixels of vertical pointer travel each simulated
            drag adds. Negative walks the pointer up the canvas, which pitches the
            camera away from the pole the lock axis points at; positive walks it down.

    Returns:
        TypeScript source that prints one JSON record for the pose the controls come up
        on and one per simulated drag, each carrying `drag`, `up_along_axis`,
        `right_along_axis`, `polar`, `written_position`, `written_up`, and
        `written_quaternion` measured on the camera the controls left behind.
    """
    return """
import { JSDOM } from "jsdom";
import * as THREE from "three";

const LOCK_ROLL = %s;
const EYE = %s;
const EYE_OFFSET_ALONG_LOCK_AXIS = %s;
const UP = %s;
const PITCH_PIXELS_PER_DRAG = %s;
const DRAG_COUNT = %s;

const dom = new JSDOM("<!doctype html><html><body></body></html>", { pretendToBeVisual: true });
const globals = globalThis as unknown as Record<string, unknown>;
globals.window = dom.window;
globals.document = dom.window.document;
globals.MutationObserver = dom.window.MutationObserver;
globals.MouseEvent = dom.window.MouseEvent;
globals.HTMLElement = dom.window.HTMLElement;

const { createTrackballCameraControls } = await import("./trackball_camera_controls.ts");

const container = dom.window.document.createElement("div");
dom.window.document.body.appendChild(container);
const canvas = dom.window.document.createElement("canvas");
container.appendChild(canvas);
// Three's own trackball captures the pointer on mousedown, which jsdom's canvas does
// not implement; the roll-locked rotation never uses it.
const canvasGlobals = canvas as unknown as Record<string, unknown>;
canvasGlobals.setPointerCapture = () => {};
canvasGlobals.releasePointerCapture = () => {};
canvasGlobals.hasPointerCapture = () => false;
const renderer = { domElement: canvas } as unknown as THREE.WebGLRenderer;

const lockRoll = new THREE.Vector3(LOCK_ROLL[0], LOCK_ROLL[1], LOCK_ROLL[2]);
const rollLockAxis = lockRoll.clone().normalize();

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
// The eye is moved along the axis the controls themselves hold, not along one computed
// elsewhere: three normalizes by multiplying by the reciprocal of the length, so an axis
// normalized any other way lands an ulp off it and a start meant to sit exactly on the
// pole sits just beside it instead.
camera.position
  .set(EYE[0], EYE[1], EYE[2])
  .addScaledVector(rollLockAxis, EYE_OFFSET_ALONG_LOCK_AXIS);
camera.up.set(UP[0], UP[1], UP[2]);
camera.lookAt(new THREE.Vector3(0, 0, 0));

const controls = createTrackballCameraControls({
  container,
  camera,
  renderer,
  initialCameraState: null,
  lockRoll,
});

function pointerEvent(type: string, clientY: number): Event {
  return new dom.window.MouseEvent(type, {
    bubbles: true,
    cancelable: true,
    button: 0,
    clientX: 0,
    clientY,
  });
}

const records: Array<Record<string, unknown>> = [];
// The camera's own basis is recorded alongside the two dot products because both of
// those read a direction and neither reads a length: three's `.normalize()` answers a
// collapsed cross product with the zero vector rather than a NaN one, so a camera left
// with a zero up vector and a non-unit quaternion - a world matrix carrying a scale,
// which renders the scene at the wrong size - passes both dot products unremarked.
function recordCameraPose(drag: number): void {
  camera.updateMatrixWorld();
  const cameraRightAxis = new THREE.Vector3()
    .setFromMatrixColumn(camera.matrixWorld, 0)
    .normalize();
  records.push({
    drag: drag,
    up_along_axis: camera.up.clone().normalize().dot(rollLockAxis),
    right_along_axis: cameraRightAxis.dot(rollLockAxis),
    polar: camera.position.clone().sub(controls.target).angleTo(rollLockAxis),
    written_position: [camera.position.x, camera.position.y, camera.position.z],
    written_up: [camera.up.x, camera.up.y, camera.up.z],
    written_quaternion: [
      camera.quaternion.x,
      camera.quaternion.y,
      camera.quaternion.z,
      camera.quaternion.w,
    ],
  });
}

recordCameraPose(0);
canvas.dispatchEvent(pointerEvent("pointerdown", 0));
for (let drag = 1; drag <= DRAG_COUNT; drag += 1) {
  dom.window.dispatchEvent(pointerEvent("pointermove", drag * PITCH_PIXELS_PER_DRAG));
  recordCameraPose(drag);
}
process.stdout.write(JSON.stringify(records));
""" % (
        json.dumps(list(lock_roll)),
        json.dumps(list(eye)),
        json.dumps(eye_offset_along_lock_axis),
        json.dumps(list(up)),
        json.dumps(pitch_pixels_per_drag),
        json.dumps(ROLL_LOCKED_PITCH_DRAG_COUNT),
    )


def run_roll_lock_pitch_harness(
    harness_dir: Path,
    lock_roll: Tuple[float, float, float],
    eye: Tuple[float, float, float],
    eye_offset_along_lock_axis: float,
    up: Tuple[float, float, float],
    pitch_pixels_per_drag: int,
) -> List[Dict[str, Any]]:
    """Lay the harness out beside the module under test and run it, reading back its records.

    The module resolves `three` from the nearest `node_modules` above it, so the harness
    directory borrows the repo's one Node package rather than the module's own location
    growing a package of its own.

    Args:
        harness_dir: Directory the harness, the copied module, and the borrowed
            `node_modules` are laid out in.
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.
        eye: The `(x, y, z)` world-space coordinates the simulated viewer's camera
            starts at, before the lock-axis offset below is added, looking at a rotation
            target on the world origin.
        eye_offset_along_lock_axis: Signed distance the eye is then moved along the
            normalized lock axis, 0 to leave the coordinates above alone.
        up: The `(x, y, z)` world-space up vector the simulated viewer's camera starts
            with.
        pitch_pixels_per_drag: Signed pixels of vertical pointer travel each simulated
            drag adds.

    Returns:
        One dict for the pose the controls come up on and one per simulated drag, each
        carrying `drag` as an int, `up_along_axis`, `right_along_axis`, and `polar` as
        floats, and `written_position`, `written_up`, and `written_quaternion` as
        component lists.
    """
    assert WEB_NODE_MODULES_PATH.is_dir(), (
        "The roll-lock harness runs the module against the repo's installed Node "
        "packages, so they must be installed. "
        f"{WEB_NODE_MODULES_PATH=}"
    )

    (harness_dir / "node_modules").symlink_to(WEB_NODE_MODULES_PATH)
    (harness_dir / "package.json").write_text('{"type": "module"}')
    (harness_dir / TRACKBALL_CAMERA_CONTROLS_SCRIPT_PATH.name).write_text(
        TRACKBALL_CAMERA_CONTROLS_SCRIPT_PATH.read_text(),
    )
    harness_path = harness_dir / "roll_lock_pitch_harness.ts"
    harness_path.write_text(
        build_roll_lock_pitch_harness_script(
            lock_roll=lock_roll,
            eye=eye,
            eye_offset_along_lock_axis=eye_offset_along_lock_axis,
            up=up,
            pitch_pixels_per_drag=pitch_pixels_per_drag,
        ),
    )

    completed_process = subprocess.run(
        args=[str(TSX_EXECUTABLE_PATH), harness_path.name],
        check=False,
        capture_output=True,
        cwd=str(harness_dir),
        text=True,
    )

    assert completed_process.returncode == 0, (
        "Expected the roll-lock pitch Node harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    records = json.loads(completed_process.stdout)
    assert len(records) == ROLL_LOCKED_PITCH_RECORD_COUNT, (
        "The harness must report one record for the pose the controls come up on and "
        f"one per simulated drag. {len(records)=} {ROLL_LOCKED_PITCH_RECORD_COUNT=}"
    )
    return records


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
    """Assert every step left the horizon level, the scene the right way up, and the camera a real camera.

    The three clauses are what a person looking at the viewer would call wrong. A camera
    right axis off the lock axis tips the horizon; an up vector on the axis's far side
    hangs the scene upside down; and a camera whose basis is not unit length is not a
    camera at all. The third is the one the first two cannot see: both read a direction
    and neither reads a length, and three answers a collapsed cross product with the
    zero vector rather than a NaN one, so a zero up vector reads as `up . axis == 0` and
    passes the upright clause while its non-unit quaternion scales the whole world
    matrix and renders the scene at the wrong size.

    Args:
        records: One dict per harness step, each carrying `drag`, `up_along_axis`,
            `right_along_axis`, `written_position`, `written_up`, and
            `written_quaternion` as `run_roll_lock_pitch_harness` returns them.

    Returns:
        None.
    """
    tilted_records = [
        record
        for record in records
        if abs(record["right_along_axis"]) > ROLL_LOCKED_PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "A roll-locked camera must keep its right axis perpendicular to the lock axis, "
        f"which is what holds the horizon level. {tilted_records=} {records=}"
    )
    inverted_records = [record for record in records if record["up_along_axis"] < 0]
    assert not inverted_records, (
        "A roll-locked camera must never hang the scene upside down, so its up vector "
        f"must stay on the lock axis's own side. {inverted_records=} {records=}"
    )
    unreal_records = [
        record
        for record in records
        if any(component is None for component in record["written_position"])
        or not is_unit_length(record["written_up"])
        or not is_unit_length(record["written_quaternion"])
    ]
    assert not unreal_records, (
        "A roll-locked camera must stay a camera a renderer can draw: a finite eye, and "
        "an up vector and orientation quaternion that are both unit length, since a "
        "non-unit quaternion scales the camera's world matrix and renders the scene at "
        f"the wrong size. {unreal_records=} {records=}"
    )


def test_the_roll_locked_rotation_stops_a_pitch_at_the_pole(tmp_path: Path) -> None:
    """A pitch that reaches the pole stops there rather than carrying the view through it, so the camera never comes out the far side."""
    records = run_roll_lock_pitch_harness(
        harness_dir=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=ROLL_LOCKED_OFF_AXIS_EYE,
        eye_offset_along_lock_axis=0.0,
        up=ROLL_LOCKED_OFF_AXIS_UP,
        pitch_pixels_per_drag=ROLL_LOCKED_PITCH_PIXELS_PER_DRAG,
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


def test_the_roll_locked_rotation_never_lets_a_pitch_invert_the_camera(
    tmp_path: Path,
) -> None:
    """A pole-reaching pitch leaves the camera up vector on the lock axis's own side at every drag, which a camera right axis perpendicular to that axis never says on its own."""
    records = run_roll_lock_pitch_harness(
        harness_dir=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=ROLL_LOCKED_OFF_AXIS_EYE,
        eye_offset_along_lock_axis=0.0,
        up=ROLL_LOCKED_OFF_AXIS_UP,
        pitch_pixels_per_drag=ROLL_LOCKED_PITCH_PIXELS_PER_DRAG,
    )

    assert_roll_locked_camera(records=records)


def test_the_roll_locked_rotation_leaves_an_eye_starting_on_the_lock_axis(
    tmp_path: Path,
) -> None:
    """A viewer framed straight down the lock axis comes up with a polar angle of 0, where the camera right axis the rotation turns about is derived from the eye offset and that axis alone, so a drag away from the pole must still move the camera and leave it drawable."""
    records = run_roll_lock_pitch_harness(
        harness_dir=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=ROLL_LOCKED_ROTATION_TARGET,
        eye_offset_along_lock_axis=ROLL_LOCKED_START_RADIUS,
        up=build_up_across_lock_axis(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        pitch_pixels_per_drag=-ROLL_LOCKED_PITCH_PIXELS_PER_DRAG,
    )

    polar_angles = [record["polar"] for record in records]
    assert max(polar_angles) >= ROLL_LOCKED_POLE_DEPARTURE_RADIANS, (
        "A camera that comes up on the pole must still be draggable off it, so a pitch "
        "away from that pole must turn the camera rather than leave it pinned where a "
        f"collapsed rotation axis put it. {polar_angles=} "
        f"{ROLL_LOCKED_POLE_DEPARTURE_RADIANS=}"
    )
    assert_roll_locked_camera(records=records)


def test_the_roll_locked_rotation_leaves_an_eye_starting_past_the_far_pole(
    tmp_path: Path,
) -> None:
    """A viewer framed straight up the lock axis comes up with a polar angle of pi, the other end of the same degeneracy, and a drag away from that pole must move the camera and leave it drawable too."""
    records = run_roll_lock_pitch_harness(
        harness_dir=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=ROLL_LOCKED_ROTATION_TARGET,
        eye_offset_along_lock_axis=-ROLL_LOCKED_START_RADIUS,
        up=build_up_across_lock_axis(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        pitch_pixels_per_drag=ROLL_LOCKED_PITCH_PIXELS_PER_DRAG,
    )

    polar_angles = [record["polar"] for record in records]
    assert min(polar_angles) <= math.pi - ROLL_LOCKED_POLE_DEPARTURE_RADIANS, (
        "A camera that comes up on the far pole must still be draggable off it, so a "
        "pitch away from that pole must turn the camera rather than leave it pinned "
        f"where a collapsed rotation axis put it. {polar_angles=} "
        f"{ROLL_LOCKED_POLE_DEPARTURE_RADIANS=}"
    )
    assert_roll_locked_camera(records=records)
