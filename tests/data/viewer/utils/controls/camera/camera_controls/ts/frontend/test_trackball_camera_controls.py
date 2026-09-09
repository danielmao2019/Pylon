"""Tests for the TypeScript trackball camera controls' roll lock."""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

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
# Radians from the lock axis at or below which the camera counts as having reached the
# pole, generous next to the controls' own stopping distance.
ROLL_LOCKED_POLE_REACHED_RADIANS = 1e-3
# Magnitude of `right . axis` at or below which the camera right axis counts as
# perpendicular to the lock axis.
ROLL_LOCKED_PERPENDICULAR_TOLERANCE = 1e-9


def build_roll_lock_pitch_harness_script(lock_roll: Tuple[float, float, float]) -> str:
    """Build the Node harness that drives the roll-locked controls through a pole-crossing pitch.

    The harness stands in for the browser: it mounts the real controls on a jsdom
    canvas over a real three camera, holds a left-drag down, and walks the pointer down
    the canvas so every drag step reaches the controls' own rotation.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.

    Returns:
        TypeScript source that prints one JSON record per simulated drag, each carrying
        `up_along_axis`, `right_along_axis`, and `polar` measured on the camera the
        controls left behind.
    """
    return """
import { JSDOM } from "jsdom";
import * as THREE from "three";

const LOCK_ROLL = %s;
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

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(0, 0, 1);
camera.up.set(0, 1, 0);
camera.lookAt(new THREE.Vector3(0, 0, 0));

const lockRoll = new THREE.Vector3(LOCK_ROLL[0], LOCK_ROLL[1], LOCK_ROLL[2]);
const rollLockAxis = lockRoll.clone().normalize();
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

canvas.dispatchEvent(pointerEvent("pointerdown", 0));
const records: Array<Record<string, number>> = [];
for (let drag = 1; drag <= DRAG_COUNT; drag += 1) {
  dom.window.dispatchEvent(pointerEvent("pointermove", drag * PITCH_PIXELS_PER_DRAG));
  camera.updateMatrixWorld();
  const cameraRightAxis = new THREE.Vector3()
    .setFromMatrixColumn(camera.matrixWorld, 0)
    .normalize();
  records.push({
    up_along_axis: camera.up.clone().normalize().dot(rollLockAxis),
    right_along_axis: cameraRightAxis.dot(rollLockAxis),
    polar: camera.position.clone().sub(controls.target).angleTo(rollLockAxis),
  });
}
process.stdout.write(JSON.stringify(records));
""" % (
        json.dumps(list(lock_roll)),
        json.dumps(ROLL_LOCKED_PITCH_PIXELS_PER_DRAG),
        json.dumps(ROLL_LOCKED_PITCH_DRAG_COUNT),
    )


def run_roll_lock_pitch_harness(
    harness_dir: Path,
    lock_roll: Tuple[float, float, float],
) -> List[Dict[str, float]]:
    """Lay the harness out beside the module under test and run it, reading back its records.

    The module resolves `three` from the nearest `node_modules` above it, so the harness
    directory borrows the repo's one Node package rather than the module's own location
    growing a package of its own.

    Args:
        harness_dir: Directory the harness, the copied module, and the borrowed
            `node_modules` are laid out in.
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.

    Returns:
        One dict per simulated drag, each carrying `up_along_axis`,
        `right_along_axis`, and `polar` as floats.
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
    harness_path.write_text(build_roll_lock_pitch_harness_script(lock_roll=lock_roll))

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
    return json.loads(completed_process.stdout)


def test_the_roll_locked_rotation_stops_a_pitch_at_the_pole(tmp_path: Path) -> None:
    """A pitch that reaches the pole stops there rather than carrying the view through it, so the camera never comes out the far side."""
    records = run_roll_lock_pitch_harness(
        harness_dir=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    assert len(records) == ROLL_LOCKED_PITCH_DRAG_COUNT, (
        "The harness must report one record per simulated drag. "
        f"{len(records)=} {ROLL_LOCKED_PITCH_DRAG_COUNT=}"
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


def test_the_roll_locked_rotation_never_lets_a_pitch_invert_the_camera(
    tmp_path: Path,
) -> None:
    """A pole-reaching pitch leaves the camera up vector on the lock axis's own side at every drag, which a camera right axis perpendicular to that axis never says on its own."""
    records = run_roll_lock_pitch_harness(
        harness_dir=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    inverted_records = [record for record in records if record["up_along_axis"] < 0]
    assert not inverted_records, (
        "A roll-locked camera must never hang the scene upside down, so its up vector "
        "must stay on the lock axis's own side through a pitch that reaches the pole. "
        f"{inverted_records=} {records=}"
    )
    tilted_records = [
        record
        for record in records
        if abs(record["right_along_axis"]) > ROLL_LOCKED_PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "A roll-locked camera must keep its right axis perpendicular to the lock axis "
        f"through the same drags. {tilted_records=} {records=}"
    )
