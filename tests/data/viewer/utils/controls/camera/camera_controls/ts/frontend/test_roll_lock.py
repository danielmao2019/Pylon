"""Tests for the geometry the three.js trackball camera controls run on a roll-locked drag."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The three.js controls are TypeScript, so the geometry under test is driven through Node rather than imported: the harness beside this test builds a jsdom document, constructs the shipped controls over a real perspective camera, and puts left-drag pointer events through the canvas.
REPO_ROOT = Path(__file__).resolve().parents[9]
ROLL_LOCK_HARNESS_SCRIPT_PATH = Path(__file__).resolve().parent / "roll_lock_harness.ts"
TSX_EXECUTABLE_PATH = REPO_ROOT / "web" / "node_modules" / ".bin" / "tsx"
# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# A lock axis the camera is seeded looking straight down, which is the one framing where the view direction runs parallel to the lock axis and the cross product that re-derives the camera right axis collapses.
TOP_DOWN_LOCK_ROLL = (0.0, 0.0, 1.0)
# Distance from the scene centre the simulated camera orbits at.
ORBIT_RADIUS = 10.0
# Magnitude of `right . axis` above which the horizon is no longer level.
PERPENDICULAR_TOLERANCE = 1e-9
# Deviation from unit length above which a reported direction is no longer a direction.
UNIT_LENGTH_TOLERANCE = 1e-9
# Radians below which two reported up vectors point the same way, so a drag that leaves them further apart turned the camera.
UP_TURNED_RADIANS = 1e-6
# Polar angle, in radians, at or below which the camera stands at the pole.
POLE_REACHED_RADIANS = 1e-3
# Pointer travel, in pixels, of one drag in each block of the pole-crossing sequence.
TURNING_DRAG = {"dx": 50, "dy": 50}
POLE_DRAG = {"dx": 0, "dy": 160}
YAW_DRAG = {"dx": 80, "dy": 0}
RETURN_DRAG = {"dx": 0, "dy": -140}
# How many drags each block of the pole-crossing sequence runs. The pole block runs well past the drag that first reaches the pole, so the sequence covers the drags a camera without the clamp spends tumbling out the far side.
TURNING_DRAG_COUNT = 4
POLE_DRAG_COUNT = 8
YAW_DRAG_COUNT = 4
RETURN_DRAG_COUNT = 4


def build_pole_crossing_drags() -> List[Dict[str, int]]:
    """Build the drag sequence that turns the camera, drives it into the pole, yaws there, and pitches back out.

    Args:
        None.

    Returns:
        One `{"dx", "dy"}` record per simulated left-drag step, in pixels of pointer travel.
    """
    return (
        [dict(TURNING_DRAG)] * TURNING_DRAG_COUNT
        + [dict(POLE_DRAG)] * POLE_DRAG_COUNT
        + [dict(YAW_DRAG)] * YAW_DRAG_COUNT
        + [dict(RETURN_DRAG)] * RETURN_DRAG_COUNT
    )


def build_equator_position(lock_roll: Tuple[float, float, float]) -> List[float]:
    """Build a camera position a lock axis's own distance from the pole, so a drag has somewhere to fall from.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.

    Returns:
        An `[x, y, z]` camera position perpendicular to the lock axis at `ORBIT_RADIUS` from the origin.
    """
    axis_length = math.sqrt(sum(component * component for component in lock_roll))
    axis = [component / axis_length for component in lock_roll]
    seed = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
    projection = sum(a * s for a, s in zip(axis, seed, strict=True))
    perpendicular = [
        seed[index] - axis[index] * projection for index in range(len(axis))
    ]
    perpendicular_length = math.sqrt(
        sum(component * component for component in perpendicular)
    )
    return [
        component / perpendicular_length * ORBIT_RADIUS for component in perpendicular
    ]


def angle_between(left: List[float], right: List[float]) -> float:
    """Compute the angle between two reported camera directions.

    Args:
        left: An `[x, y, z]` direction in the scene's own world frame.
        right: An `[x, y, z]` direction in the scene's own world frame.

    Returns:
        The angle between the two directions, in radians.
    """
    left_length = math.sqrt(sum(component * component for component in left))
    right_length = math.sqrt(sum(component * component for component in right))
    cosine = (
        sum(a * b for a, b in zip(left, right, strict=True))
        / left_length
        / right_length
    )
    return math.acos(max(-1.0, min(1.0, cosine)))


def write_harness_module_resolution_config(directory: Path) -> Path:
    """Write the TypeScript config mapping the bare module specifiers the harness and the controls import onto this checkout.

    Args:
        directory: Directory the config is written into.

    Returns:
        Path the config was written to.
    """
    config_path = directory / "tsconfig.json"
    config_path.write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "target": "ES2020",
                    "module": "ESNext",
                    "moduleResolution": "node",
                    "baseUrl": str(REPO_ROOT),
                    "paths": {
                        "three": ["web/node_modules/three/build/three.module.js"],
                        "three/*": ["web/node_modules/three/*"],
                        "jsdom": ["web/node_modules/jsdom/lib/api.js"],
                        "data/*": ["data/*"],
                    },
                }
            }
        )
    )
    return config_path


def run_roll_lock_harness(
    tmp_path: Path,
    lock_roll: Optional[Tuple[float, float, float]],
    position: List[float],
    up: List[float],
    drags: List[Dict[str, int]],
) -> Dict[str, Any]:
    """Drive the shipped trackball camera controls through a scripted left-drag under Node and read back the camera they left behind.

    Args:
        tmp_path: Directory the generated TypeScript module-resolution config is written into.
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length; None constructs the free trackball.
        position: The `[x, y, z]` camera position the controls are seeded with, in the scene's own world frame.
        up: The `[x, y, z]` camera up vector the controls are seeded with, in the scene's own world frame.
        drags: One `{"dx", "dy"}` record per simulated left-drag step, in pixels of pointer travel.

    Returns:
        A dict carrying `roll_lock_axis`, `roll_lock_polar_angle_epsilon`, and `records` -- one record for the constructed camera followed by one per drag step, each carrying `right_along_axis`, `up_along_axis`, `up_length`, `camera_right_axis_length`, `polar`, `position`, `up`, `camera_right_axis`, and `finite`.
    """
    assert ROLL_LOCK_HARNESS_SCRIPT_PATH.is_file(), (
        "The roll-lock TypeScript harness must sit beside this test. ROLL_LOCK_HARNESS_SCRIPT_PATH=%r"
        % (ROLL_LOCK_HARNESS_SCRIPT_PATH,)
    )
    assert TSX_EXECUTABLE_PATH.is_file(), (
        "The TypeScript runner the harness is driven with must be installed; run `npm install` in the web workspace. "
        "TSX_EXECUTABLE_PATH=%r" % (TSX_EXECUTABLE_PATH,)
    )

    completed_process = subprocess.run(
        args=[
            str(TSX_EXECUTABLE_PATH),
            "--tsconfig",
            str(write_harness_module_resolution_config(directory=tmp_path)),
            str(ROLL_LOCK_HARNESS_SCRIPT_PATH),
            json.dumps(
                {
                    "lock_roll": None if lock_roll is None else list(lock_roll),
                    "position": position,
                    "target": [0.0, 0.0, 0.0],
                    "up": up,
                    "drags": drags,
                }
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert completed_process.returncode == 0, (
        "Expected the roll-lock TypeScript harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    payload = json.loads(completed_process.stdout)
    assert len(payload["records"]) == len(drags) + 1, (
        "The harness must report the constructed camera and one camera per drag step. "
        f"{len(payload['records'])=} {len(drags)=}"
    )
    return payload


def test_a_camera_looking_down_the_lock_axis_keeps_a_usable_frame(
    tmp_path: Path,
) -> None:
    """A camera seeded looking straight down the lock axis is left with a real camera frame and still turns, rather than collapsing onto the direction the vanished cross product between the view direction and the lock axis cannot define.

    Args:
        tmp_path: Temporary directory for the generated module-resolution config.

    Returns:
        None.
    """
    payload = run_roll_lock_harness(
        tmp_path=tmp_path,
        lock_roll=TOP_DOWN_LOCK_ROLL,
        position=[0.0, 0.0, ORBIT_RADIUS],
        up=list(TOP_DOWN_LOCK_ROLL),
        drags=[dict(RETURN_DRAG)] * RETURN_DRAG_COUNT,
    )

    records = payload["records"]
    unusable_records = [
        record
        for record in records
        if not record["finite"]
        or not abs(record["up_length"] - 1.0) <= UNIT_LENGTH_TOLERANCE
        or not abs(record["camera_right_axis_length"] - 1.0) <= UNIT_LENGTH_TOLERANCE
    ]
    assert not unusable_records, (
        "A roll-locked camera must be left with unit-length, finite up and right axes whatever framing it was seeded with, so the renderer is never handed a pose it cannot draw. "
        f"{unusable_records=} {UNIT_LENGTH_TOLERANCE=}"
    )
    tilted_records = [
        record
        for record in records
        if not abs(record["right_along_axis"]) <= PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "The same camera must keep its right axis perpendicular to the lock axis. "
        f"{tilted_records=} {PERPENDICULAR_TOLERANCE=}"
    )
    inverted_records = [
        record for record in records if not record["up_along_axis"] >= 0
    ]
    assert not inverted_records, (
        "The same camera must keep its up vector on the lock axis's own side. "
        f"{inverted_records=}"
    )
    departed_polar_angle = records[-1]["polar"]
    assert departed_polar_angle > POLE_REACHED_RADIANS, (
        "A pitch away from the pole must carry a camera seeded on the lock axis off it, so the seeding never traps the camera there. "
        f"{departed_polar_angle=} {POLE_REACHED_RADIANS=}"
    )


def test_a_pole_crossing_drag_holds_the_horizon_level(tmp_path: Path) -> None:
    """The camera right axis stays perpendicular to the lock axis through every step of a drag that turns the camera, drives it into the pole, yaws there, and pitches back out.

    Args:
        tmp_path: Temporary directory for the generated module-resolution config.

    Returns:
        None.
    """
    payload = run_roll_lock_harness(
        tmp_path=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        position=build_equator_position(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        drags=build_pole_crossing_drags(),
    )

    tilted_records = [
        record
        for record in payload["records"]
        if not abs(record["right_along_axis"]) <= PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "A roll-locked camera must keep its right axis perpendicular to the lock axis, so the horizon stays level. "
        f"{tilted_records=} {PERPENDICULAR_TOLERANCE=}"
    )


def test_a_pole_crossing_drag_never_hangs_the_scene_upside_down(tmp_path: Path) -> None:
    """The camera up vector stays on the lock axis's own side through the same drag, which a camera right axis perpendicular to that axis never says on its own.

    Args:
        tmp_path: Temporary directory for the generated module-resolution config.

    Returns:
        None.
    """
    payload = run_roll_lock_harness(
        tmp_path=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        position=build_equator_position(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        drags=build_pole_crossing_drags(),
    )

    inverted_records = [
        record for record in payload["records"] if not record["up_along_axis"] >= 0
    ]
    assert not inverted_records, (
        "A roll-locked camera must never hang the scene upside down, so its up vector must stay on the lock axis's own side. "
        f"{inverted_records=}"
    )


def test_the_pitch_clamp_leaves_the_camera_turning(tmp_path: Path) -> None:
    """The drag steps that follow a camera parked at the pole still turn it and still pitch it back out, so stopping the pitch at the pole never froze the camera in place.

    Args:
        tmp_path: Temporary directory for the generated module-resolution config.

    Returns:
        None.
    """
    payload = run_roll_lock_harness(
        tmp_path=tmp_path,
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        position=build_equator_position(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        drags=build_pole_crossing_drags(),
    )

    records = payload["records"]
    parked_polar_angle = records[TURNING_DRAG_COUNT + POLE_DRAG_COUNT]["polar"]
    assert parked_polar_angle <= POLE_REACHED_RADIANS, (
        "The pitch must actually park the camera at the pole, or nothing about turning away from it is under test. "
        f"{parked_polar_angle=} {POLE_REACHED_RADIANS=}"
    )
    yaw_records = records[
        TURNING_DRAG_COUNT
        + POLE_DRAG_COUNT : TURNING_DRAG_COUNT
        + POLE_DRAG_COUNT
        + YAW_DRAG_COUNT
        + 1
    ]
    still_records = [
        (before, after)
        for before, after in zip(yaw_records[:-1], yaw_records[1:], strict=True)
        if not angle_between(before["up"], after["up"]) > UP_TURNED_RADIANS
    ]
    assert not still_records, (
        "Every yaw drag step from the pole must turn the camera, so the clamp that stops the pitch never froze the yaw. "
        f"{still_records=} {UP_TURNED_RADIANS=}"
    )
    returned_polar_angle = records[-1]["polar"]
    assert returned_polar_angle > POLE_REACHED_RADIANS, (
        "A reversed pitch must carry the camera back off the pole, so the clamp holds the camera there rather than trapping it. "
        f"{returned_polar_angle=} {POLE_REACHED_RADIANS=}"
    )


def test_the_unlocked_path_buys_no_roll_lock(tmp_path: Path) -> None:
    """Controls built with no lock axis expose neither the axis nor the pitch clamp and never park the camera in the clamp's polar band, so the same drag leaves the free trackball exactly as three constructed it.

    Args:
        tmp_path: Temporary directory for the generated module-resolution config.

    Returns:
        None.
    """
    payload = run_roll_lock_harness(
        tmp_path=tmp_path,
        lock_roll=None,
        position=build_equator_position(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        drags=build_pole_crossing_drags(),
    )

    assert payload["roll_lock_axis"] is None, (
        "Free trackball camera controls must own no lock axis. "
        f"{payload['roll_lock_axis']=}"
    )
    assert payload["roll_lock_polar_angle_epsilon"] is None, (
        "Free trackball camera controls must own no pitch clamp. "
        f"{payload['roll_lock_polar_angle_epsilon']=}"
    )
    clamped_records = [
        record
        for record in payload["records"]
        if record["polar"] <= POLE_REACHED_RADIANS
    ]
    assert not clamped_records, (
        "A free trackball must never park the camera in the roll-locked pitch clamp's polar band, which is where the roll-locked drag stops it. "
        f"{clamped_records=} {POLE_REACHED_RADIANS=}"
    )
