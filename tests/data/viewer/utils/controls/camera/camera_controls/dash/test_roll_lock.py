"""Tests for the geometry the Dash roll-lock clientside callback runs on a gl3d camera."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCK_CALLBACK_SCRIPT_PATH,
)

# The Node harness standing in for the Plotly panel the callback corrects.
ROLL_LOCK_HARNESS_SCRIPT_PATH = Path(__file__).resolve().parent / "roll_lock_harness.js"
# The component id the roll-locked graph is registered under.
ROLL_LOCKED_GRAPH_ID = "roll-locked-graph"
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
# Distance below which two reported eye positions are the same position, so a drag that leaves them apart by more than this moved the camera.
EYE_MOVED_DISTANCE = 1e-6
# Polar angle, in radians, at or below which the camera stands at the pole.
POLE_REACHED_RADIANS = 1e-3
# Yaw and pitch, in radians, of one turn in each block of the pole-crossing sequence.
TURNING_DRAG = {"yaw": 0.30, "pitch": -0.20}
POLE_DRAG = {"yaw": 0.0, "pitch": -0.55}
YAW_DRAG = {"yaw": 0.45, "pitch": 0.0}
RETURN_DRAG = {"yaw": 0.0, "pitch": 0.55}
# How many turns each block of the pole-crossing sequence runs. The pole block runs well past the turn that first reaches the pole, so the sequence covers the turns a camera without the clamp spends tumbling out the far side.
TURNING_DRAG_COUNT = 4
POLE_DRAG_COUNT = 8
YAW_DRAG_COUNT = 4
RETURN_DRAG_COUNT = 4
# Yaw and pitch, in radians, of one pointer move of the live drag, sized so the whole run sweeps the camera well off its start without any single move jumping it there.
LIVE_DRAG_MOVE = {"yaw": 0.06, "pitch": -0.045}
# How many pointer moves the live drag runs, which is how many the panel reports nothing of.
LIVE_DRAG_MOVE_COUNT = 24


def build_pole_crossing_drags() -> List[Dict[str, float]]:
    """Build the drag sequence that turns the camera, drives it into the pole, yaws there, and pitches back out.

    Args:
        None.

    Returns:
        One `{"yaw", "pitch"}` record per simulated `orbit` left-drag, in radians.
    """
    return (
        [dict(TURNING_DRAG)] * TURNING_DRAG_COUNT
        + [dict(POLE_DRAG)] * POLE_DRAG_COUNT
        + [dict(YAW_DRAG)] * YAW_DRAG_COUNT
        + [dict(RETURN_DRAG)] * RETURN_DRAG_COUNT
    )


def build_equator_eye(lock_roll: Tuple[float, float, float]) -> List[float]:
    """Build an eye position a lock axis's own distance from the pole, so a drag has somewhere to fall from.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.

    Returns:
        An `[x, y, z]` eye position perpendicular to the lock axis at `ORBIT_RADIUS` from the origin.
    """
    axis_length = math.sqrt(sum(component * component for component in lock_roll))
    axis = [component / axis_length for component in lock_roll]
    seed = [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]
    perpendicular = [
        seed[0] - axis[0] * sum(a * s for a, s in zip(axis, seed, strict=True)),
        seed[1] - axis[1] * sum(a * s for a, s in zip(axis, seed, strict=True)),
        seed[2] - axis[2] * sum(a * s for a, s in zip(axis, seed, strict=True)),
    ]
    perpendicular_length = math.sqrt(
        sum(component * component for component in perpendicular)
    )
    return [
        component / perpendicular_length * ORBIT_RADIUS for component in perpendicular
    ]


def run_roll_lock_harness(
    lock_roll: Tuple[float, float, float],
    eye: List[float],
    up: List[float],
    turns: List[Dict[str, float]],
    reports_each_turn: bool = True,
) -> List[Dict[str, Any]]:
    """Drive the shipped roll-lock callback through a scripted gl3d drag under Node and read back the camera it left behind.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.
        eye: The `[x, y, z]` eye position the panel is seeded with, in the scene's own world frame.
        up: The `[x, y, z]` camera up vector the panel is seeded with, in the scene's own world frame.
        turns: One `{"yaw", "pitch"}` record per simulated `orbit` rotation of the camera, in radians.
        reports_each_turn: The cadence the panel reports those turns to the callback at. True makes each turn a drag of its own, reported at the mouse-up gl3d emits `plotly_relayout` on; False makes the turns the pointer moves of one live drag, which the panel reports nothing of until it ends.

    Returns:
        One record for the seeded camera followed by one per turn, each carrying `right_along_axis`, `up_along_axis`, `up_length`, `camera_right_axis_length`, `polar`, `eye`, `up`, `camera_right_axis`, and `finite`.
    """
    assert ROLL_LOCK_HARNESS_SCRIPT_PATH.is_file(), (
        "The roll-lock Node harness must sit beside this test. ROLL_LOCK_HARNESS_SCRIPT_PATH=%r"
        % (ROLL_LOCK_HARNESS_SCRIPT_PATH,)
    )

    completed_process = subprocess.run(
        args=[
            "node",
            str(ROLL_LOCK_HARNESS_SCRIPT_PATH),
            json.dumps(
                {
                    "source_path": str(ROLL_LOCK_CALLBACK_SCRIPT_PATH),
                    "graph_id": ROLL_LOCKED_GRAPH_ID,
                    "lock_roll": list(lock_roll),
                    "eye": eye,
                    "center": [0.0, 0.0, 0.0],
                    "up": up,
                    "turns": turns,
                    "reports_each_turn": reports_each_turn,
                }
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, (
        "Expected the roll-lock Node harness to succeed. "
        f"{completed_process.returncode=} {completed_process.stderr=}"
    )
    records = json.loads(completed_process.stdout)
    assert len(records) == len(turns) + 1, (
        "The harness must report the seeded camera and one camera per turn. "
        f"{len(records)=} {len(turns)=}"
    )
    return records


def test_a_camera_looking_down_the_lock_axis_keeps_a_usable_frame() -> None:
    """A camera seeded looking straight down the lock axis is left with a real camera frame, rather than the direction the collapsed cross product between the view direction and the lock axis cannot define.

    Args:
        None.

    Returns:
        None.
    """
    records = run_roll_lock_harness(
        lock_roll=TOP_DOWN_LOCK_ROLL,
        eye=[0.0, 0.0, ORBIT_RADIUS],
        up=list(TOP_DOWN_LOCK_ROLL),
        turns=[],
    )

    unusable_records = [
        record
        for record in records
        if not record["finite"]
        or not abs(record["up_length"] - 1.0) <= UNIT_LENGTH_TOLERANCE
        or not abs(record["camera_right_axis_length"] - 1.0) <= UNIT_LENGTH_TOLERANCE
    ]
    assert not unusable_records, (
        "A roll-locked camera must be left with unit-length, finite up and right axes whatever framing the panel reports, so the panel is never handed a pose it cannot render. "
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


def test_a_pole_crossing_drag_holds_the_horizon_level() -> None:
    """The camera right axis stays perpendicular to the lock axis through every drag of a sequence that turns the camera, drives it into the pole, yaws there, and pitches back out.

    Args:
        None.

    Returns:
        None.
    """
    records = run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=build_equator_eye(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        turns=build_pole_crossing_drags(),
    )

    tilted_records = [
        record
        for record in records
        if not abs(record["right_along_axis"]) <= PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "A roll-locked camera must keep its right axis perpendicular to the lock axis, so the horizon stays level. "
        f"{tilted_records=} {PERPENDICULAR_TOLERANCE=}"
    )


def test_a_pole_crossing_drag_never_hangs_the_scene_upside_down() -> None:
    """The camera up vector stays on the lock axis's own side through the same sequence, which a camera right axis perpendicular to that axis never says on its own.

    Args:
        None.

    Returns:
        None.
    """
    records = run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=build_equator_eye(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        turns=build_pole_crossing_drags(),
    )

    inverted_records = [
        record for record in records if not record["up_along_axis"] >= 0
    ]
    assert not inverted_records, (
        "A roll-locked camera must never hang the scene upside down, so its up vector must stay on the lock axis's own side. "
        f"{inverted_records=}"
    )


def test_the_pole_clamp_leaves_the_camera_turning() -> None:
    """The drags that follow a camera parked at the pole still move it, so stopping the pitch at the pole never froze the camera in place.

    Args:
        None.

    Returns:
        None.
    """
    records = run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=build_equator_eye(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        turns=build_pole_crossing_drags(),
    )

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
        if math.dist(before["eye"], after["eye"]) <= EYE_MOVED_DISTANCE
    ]
    assert not still_records, (
        "Every yaw drag from the pole must move the camera, so the clamp that stops the pitch never froze the yaw. "
        f"{still_records=} {EYE_MOVED_DISTANCE=}"
    )


def run_live_drag() -> List[Dict[str, Any]]:
    """Drive one live `orbit` left-drag past the callback, which the panel reports nothing of until it ends, and read back the camera at every pointer move of it.

    Args:
        None.

    Returns:
        One record for the seeded camera followed by one per pointer move, as `run_roll_lock_harness` returns them.
    """
    return run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=build_equator_eye(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        turns=[dict(LIVE_DRAG_MOVE)] * LIVE_DRAG_MOVE_COUNT,
        reports_each_turn=False,
    )


def test_a_live_drag_holds_the_horizon_level_at_every_pointer_move() -> None:
    """The camera right axis stays perpendicular to the lock axis at every pointer move of a drag the panel reports nothing of, so the horizon is level under the pointer and not only once the button comes up.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_drag()

    tilted_records = [
        record
        for record in records
        if not abs(record["right_along_axis"]) <= PERPENDICULAR_TOLERANCE
    ]
    assert not tilted_records, (
        "A roll-locked camera must keep its right axis perpendicular to the lock axis at every pointer move of a live drag, since a panel that reports its camera only at mouse-up rolls under the pointer for the whole of the drag and re-levels on release. "
        f"{tilted_records=} {PERPENDICULAR_TOLERANCE=}"
    )


def test_a_live_drag_keeps_the_camera_turning_at_every_pointer_move() -> None:
    """Every pointer move of that same drag moves the camera, so holding the horizon level through the drag never froze it under the pointer.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_drag()

    still_records = [
        (before, after)
        for before, after in zip(records[:-1], records[1:], strict=True)
        if math.dist(before["eye"], after["eye"]) <= EYE_MOVED_DISTANCE
    ]
    assert not still_records, (
        "Every pointer move of a live drag must move the camera, so a roll lock that holds the horizon level through the drag by pinning the camera in place is caught here rather than read as a lock. "
        f"{still_records=} {EYE_MOVED_DISTANCE=}"
    )
