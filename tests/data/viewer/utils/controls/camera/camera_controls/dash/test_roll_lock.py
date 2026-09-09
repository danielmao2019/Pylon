"""Tests for the roll lock the Dash clientside callback holds a shipped gl3d view controller to."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCK_CALLBACK_SCRIPT_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[8]
# The Node harness standing in for the browser the shipped gl3d view controller runs in.
ROLL_LOCK_HARNESS_SCRIPT_PATH = Path(__file__).resolve().parent / "roll_lock_harness.js"
# The harness resolves `jsdom` and the shipped `plotly.js` release out of this tree, so a checkout that has not run `npm install` under `web` cannot run these tests.
NODE_MODULES_PATH = REPO_ROOT / "web" / "node_modules"
# The gl3d view controller under the harness comes out of this release, and Dash serves the bundle built from it, so the two must be the same release for the harness to be driving what ships.
HARNESS_PLOTLY_BUNDLE_PATH = NODE_MODULES_PATH / "plotly.js" / "dist" / "plotly.min.js"
SERVED_PLOTLY_BUNDLE_PATH = (
    Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
)
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
# Pointer travel, in pixels, of one drag in each block of the pole-crossing sequence.
TURNING_DRAG = {"dx": 28, "dy": 19}
POLE_DRAG = {"dx": 0, "dy": 52}
YAW_DRAG = {"dx": 43, "dy": 0}
RETURN_DRAG = {"dx": 0, "dy": -52}
# How many drags each block of the pole-crossing sequence runs. The pole block runs well past the drag that first reaches the pole, so the sequence covers the drags a camera without the clamp spends tumbling out the far side.
TURNING_DRAG_COUNT = 4
POLE_DRAG_COUNT = 8
YAW_DRAG_COUNT = 4
RETURN_DRAG_COUNT = 4
# Pointer travel, in pixels, of one pointer move of the live drag, sized so the whole run sweeps the camera well off its start without any single move jumping it there.
LIVE_DRAG_MOVE = {"dx": 6, "dy": -4}
# How many pointer moves the live drag runs, which is how many the panel reports nothing of.
LIVE_DRAG_MOVE_COUNT = 24
# Pointer travel, in pixels, of one pointer move of the live drag that pitches the camera into the pole and keeps pushing past it. A pure-vertical drag introduces no roll of its own, so the horizon reads the same under a locked and an unlocked panel and only the up vector's side of the lock axis separates them.
LIVE_POLE_MOVE = {"dx": 0, "dy": 26}
# How many pointer moves that drag runs, which carries it well past the move that first reaches the pole.
LIVE_POLE_MOVE_COUNT = 24


def build_pole_crossing_drags() -> List[Dict[str, int]]:
    """Build the drag sequence that turns the camera, drives it into the pole, yaws there, and pitches back out.

    Args:
        None.

    Returns:
        One `{"dx", "dy"}` record per simulated `orbit` left-drag, in pixels of pointer travel.
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


def run_roll_lock_harness(
    lock_roll: Tuple[float, float, float],
    eye: List[float],
    up: List[float],
    drags: List[Dict[str, int]],
    reports_each_drag: bool = True,
) -> List[Dict[str, Any]]:
    """Drive the shipped roll-lock callback over the shipped gl3d view controller through a scripted drag under Node and read back the camera each rendered frame drew.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.
        eye: The `[x, y, z]` eye position the panel is seeded with, in the scene's own world frame.
        up: The `[x, y, z]` camera up vector the panel is seeded with, in the scene's own world frame.
        drags: One `{"dx", "dy"}` record per simulated `orbit` pointer move, in pixels of pointer travel.
        reports_each_drag: The cadence the panel reports those moves to the callback at. True makes each move a drag of its own, released at the mouse-up gl3d emits `plotly_relayout` on and reported there; False makes them the pointer moves of one live drag, which the panel reports nothing of until the button comes up.

    Returns:
        One record for the seeded camera followed by one per pointer move, each carrying `right_along_axis`, `up_along_axis`, `up_length`, `camera_right_axis_length`, `polar`, `eye`, `up`, `camera_right_axis`, and `finite`.
    """
    assert ROLL_LOCK_HARNESS_SCRIPT_PATH.is_file(), (
        "The roll-lock Node harness must sit beside this test. ROLL_LOCK_HARNESS_SCRIPT_PATH=%r"
        % (ROLL_LOCK_HARNESS_SCRIPT_PATH,)
    )
    assert NODE_MODULES_PATH.is_dir(), (
        "The harness runs the shipped gl3d view controller out of the web workspace's installed packages, so `npm install` must have run under `web`. "
        f"{NODE_MODULES_PATH=}"
    )

    completed_process = subprocess.run(
        args=[
            "node",
            str(ROLL_LOCK_HARNESS_SCRIPT_PATH),
            json.dumps(
                {
                    "node_modules_path": str(NODE_MODULES_PATH),
                    "source_path": str(ROLL_LOCK_CALLBACK_SCRIPT_PATH),
                    "graph_id": ROLL_LOCKED_GRAPH_ID,
                    "lock_roll": list(lock_roll),
                    "eye": eye,
                    "center": [0.0, 0.0, 0.0],
                    "up": up,
                    "drags": drags,
                    "reports_each_drag": reports_each_drag,
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
    assert len(records) == len(drags) + 1, (
        "The harness must report the seeded camera and one camera per pointer move. "
        f"{len(records)=} {len(drags)=}"
    )
    return records


def test_the_harness_runs_the_view_controller_the_app_serves() -> None:
    """The `plotly.js` release the harness takes its gl3d view controller from is the release Dash serves the panel, so the spline, the idle and the recalc under test are the ones that ship rather than a differently versioned fork of them.

    Args:
        None.

    Returns:
        None.
    """
    assert HARNESS_PLOTLY_BUNDLE_PATH.is_file(), (
        "The harness's `plotly.js` release must be installed under the web workspace, so `npm install` must have run under `web`. "
        f"{HARNESS_PLOTLY_BUNDLE_PATH=}"
    )
    assert SERVED_PLOTLY_BUNDLE_PATH.is_file(), (
        "Dash serves the panel the bundle the installed `plotly` distribution carries, so that bundle must be on disk. "
        f"{SERVED_PLOTLY_BUNDLE_PATH=}"
    )

    harness_bundle = HARNESS_PLOTLY_BUNDLE_PATH.read_bytes()
    served_bundle = SERVED_PLOTLY_BUNDLE_PATH.read_bytes()
    assert harness_bundle == served_bundle, (
        "The harness must drive the same `plotly.js` release the app serves, or it certifies a view controller nobody runs. Pin the `plotly.js` version in `web/package.json` to the one the installed `plotly` distribution carries. "
        f"{HARNESS_PLOTLY_BUNDLE_PATH=} {len(harness_bundle)=} {SERVED_PLOTLY_BUNDLE_PATH=} {len(served_bundle)=}"
    )


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
        drags=[],
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
        drags=build_pole_crossing_drags(),
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
        drags=build_pole_crossing_drags(),
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
        drags=build_pole_crossing_drags(),
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
    """Drive one live `orbit` left-drag past the callback, which the panel reports nothing of until it ends, and read back the camera every rendered frame of it drew.

    Args:
        None.

    Returns:
        One record for the seeded camera followed by one per pointer move, as `run_roll_lock_harness` returns them.
    """
    return run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=build_equator_eye(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        drags=[dict(LIVE_DRAG_MOVE)] * LIVE_DRAG_MOVE_COUNT,
        reports_each_drag=False,
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


def test_a_live_drag_never_hangs_the_scene_upside_down() -> None:
    """The camera up vector stays on the lock axis's own side at every pointer move of a live pure-vertical drag that pushes well past the pole, which is the half of the lock a level horizon never says on its own and the only half a drag introducing no roll can be read by.

    Args:
        None.

    Returns:
        None.
    """
    records = run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=build_equator_eye(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
        up=list(NON_AXIS_ALIGNED_LOCK_ROLL),
        drags=[dict(LIVE_POLE_MOVE)] * LIVE_POLE_MOVE_COUNT,
        reports_each_drag=False,
    )

    reached_pole_records = [
        record for record in records if record["polar"] <= POLE_REACHED_RADIANS
    ]
    assert reached_pole_records, (
        "The drag must actually carry the camera to the pole, or nothing about being held on the lock axis's own side is under test. "
        f"{[record['polar'] for record in records]=} {POLE_REACHED_RADIANS=}"
    )
    inverted_records = [
        record for record in records if not record["up_along_axis"] >= 0
    ]
    assert not inverted_records, (
        "A roll-locked camera must never hang the scene upside down at any pointer move of a live drag, since a panel that reports its camera only at mouse-up tumbles out the far side of the pole under the pointer and rights itself on release. "
        f"{inverted_records=}"
    )
