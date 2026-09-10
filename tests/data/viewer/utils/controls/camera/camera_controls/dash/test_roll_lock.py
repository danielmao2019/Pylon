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

# The name a repository root carries and nothing above or below it does, which is what the root is found by.
REPO_ROOT_MARKER = ".git"


def resolve_repo_root() -> Path:
    """Resolve the repository root as the nearest ancestor of this file carrying the repository marker.

    Args:
        None.

    Returns:
        The path of that ancestor directory.
    """
    test_file_path = Path(__file__).resolve()
    marked_ancestors = [
        directory
        for directory in test_file_path.parents
        if (directory / REPO_ROOT_MARKER).exists()
    ]
    assert marked_ancestors, (
        "This test reads the web workspace out of the repository root, and no ancestor of this file carries the repository marker, so there is no root to read it out of. "
        f"{REPO_ROOT_MARKER=} {test_file_path=}"
    )
    return marked_ancestors[0]


REPO_ROOT = resolve_repo_root()
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
# The camera eye a panel whose framing is not degenerate is seeded with, off the lock axis so a pitch reaches a pole from a frame that is not already on one.
OFF_AXIS_EYE = (1.25, 1.25, 1.25)
# Distance from the rotation target the degenerate framings seed the eye at, matching the off-axis framing's own radius so every framing turns through the same sphere.
SEEDED_FRAMING_RADIUS = 1.25 * math.sqrt(3.0)
# Pointer travel, in pixels, of one pointer move of the live pitch that climbs into one pole, and of the reversed pitch that carries the camera back through the sphere into the other. A pure-vertical drag composes no roll of its own, so what these moves carry into the poles is the framing the panel was seeded with.
LIVE_PITCH_MOVE = {"dx": 0, "dy": -26}
REVERSED_LIVE_PITCH_MOVE = {"dx": 0, "dy": 26}
# How many pointer moves each half of that pitch runs, enough to reach its own pole from every framing below and to keep pushing well past the move that first reaches it.
LIVE_PITCH_MOVE_COUNT = 16


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


def build_live_pitch_moves() -> List[Dict[str, int]]:
    """Build the pointer moves that pitch the camera into one pole and on back through the sphere into the other.

    Args:
        None.

    Returns:
        One `{"dx", "dy"}` record per simulated `orbit` pointer move, in pixels of pointer travel.
    """
    return [dict(LIVE_PITCH_MOVE)] * LIVE_PITCH_MOVE_COUNT + [
        dict(REVERSED_LIVE_PITCH_MOVE)
    ] * LIVE_PITCH_MOVE_COUNT


def normalize_vector(vector: List[float]) -> List[float]:
    """Scale a non-zero world-space vector to unit length.

    Args:
        vector: Non-zero `[x, y, z]` world-space vector of any length.

    Returns:
        The `[x, y, z]` components at unit length.
    """
    length = math.sqrt(sum(component * component for component in vector))
    assert length > 0, f"Cannot normalize a zero-length vector. {vector=}"
    return [component / length for component in vector]


def cross_vectors(left: List[float], right: List[float]) -> List[float]:
    """Compute the cross product of two world-space vectors.

    Args:
        left: The `[x, y, z]` world-space vector on the left of the product.
        right: The `[x, y, z]` world-space vector on the right of the product.

    Returns:
        The product's `[x, y, z]` components.
    """
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def build_eye_along_lock_axis(
    lock_roll: Tuple[float, float, float],
    radius: float,
) -> List[float]:
    """Build the camera eye sitting on the lock axis itself, where the roll lock's own polar angle is degenerate.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.
        radius: Signed distance from the rotation target along the normalized axis. Positive puts the eye on the pole the axis points at, so the polar angle is 0; negative puts it past the far pole, so the polar angle is pi.

    Returns:
        An `[x, y, z]` eye position in the scene's own world frame.
    """
    axis = normalize_vector(vector=list(lock_roll))
    return [component * radius for component in axis]


def build_up_across_lock_axis(lock_roll: Tuple[float, float, float]) -> List[float]:
    """Build a camera up vector perpendicular to the lock axis, which is what keeps a camera framed straight down that axis drawable.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.

    Returns:
        The up vector's unit-length `[x, y, z]` components.
    """
    axis = normalize_vector(vector=list(lock_roll))
    least_aligned_basis = min(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        key=lambda basis: abs(
            sum(
                axis_component * basis_component
                for axis_component, basis_component in zip(axis, basis, strict=True)
            ),
        ),
    )
    return normalize_vector(vector=cross_vectors(left=axis, right=least_aligned_basis))


def build_inverted_up(
    lock_roll: Tuple[float, float, float],
    eye: List[float],
) -> List[float]:
    """Build the camera up vector a panel reports once a drag has carried the view through a pole, which is the roll-locked up vector hanging on the axis's far side.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)` world-space direction of any length.
        eye: The `[x, y, z]` eye position, off the lock axis, looking at a rotation target on the world origin.

    Returns:
        The up vector's unit-length `[x, y, z]` components, perpendicular to the view direction as every up vector a gl3d panel reports is.
    """
    axis = normalize_vector(vector=list(lock_roll))
    forward = normalize_vector(vector=[-eye[0], -eye[1], -eye[2]])
    right = normalize_vector(vector=cross_vectors(left=forward, right=axis))
    up = normalize_vector(vector=cross_vectors(left=right, right=forward))
    return [-up[0], -up[1], -up[2]]


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


def assert_roll_locked_camera(records: List[Dict[str, Any]]) -> None:
    """Assert every frame drew a camera a renderer can draw, level about the lock axis and the right way up.

    Args:
        records: One record per rendered camera, as `run_roll_lock_harness` returns them.

    Returns:
        None.
    """
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
        "A roll-locked camera must keep its right axis perpendicular to the lock axis, so the horizon stays level. "
        f"{tilted_records=} {PERPENDICULAR_TOLERANCE=}"
    )
    inverted_records = [
        record for record in records if not record["up_along_axis"] >= 0
    ]
    assert not inverted_records, (
        "A roll-locked camera must keep its up vector on the lock axis's own side, so the scene never hangs upside down. "
        f"{inverted_records=}"
    )


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

    assert_roll_locked_camera(records=records)


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


def run_live_pitch(eye: List[float], up: List[float]) -> List[Dict[str, Any]]:
    """Drive one live `orbit` pitch through both poles from a seeded framing and read back the camera every rendered frame of it drew.

    The panel reports the framing it was seeded with on its initial render and then nothing until the button comes up, so every frame between those two is held by the wrapped rotation alone.

    Args:
        eye: The `[x, y, z]` eye position the panel is seeded with, in the scene's own world frame.
        up: The `[x, y, z]` camera up vector the panel is seeded with, in the scene's own world frame.

    Returns:
        One record for the seeded camera followed by one per pointer move, as `run_roll_lock_harness` returns them.
    """
    return run_roll_lock_harness(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        eye=eye,
        up=up,
        drags=build_live_pitch_moves(),
        reports_each_drag=False,
    )


def test_the_roll_lock_callback_stops_a_pitch_at_the_pole() -> None:
    """A pitch pushed well past a pole leaves the camera standing at that pole rather than carrying the view out the far side, at both ends of the lock axis.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_pitch(
        eye=list(OFF_AXIS_EYE),
        up=normalize_vector(vector=list(NON_AXIS_ALIGNED_LOCK_ROLL)),
    )

    polar_angles = [record["polar"] for record in records]
    assert polar_angles[LIVE_PITCH_MOVE_COUNT] >= math.pi - POLE_REACHED_RADIANS, (
        "The moves that climb into the far pole must leave the camera standing at it, since they push well past the move that first reaches it. "
        f"{polar_angles=} {POLE_REACHED_RADIANS=}"
    )
    assert polar_angles[-1] <= POLE_REACHED_RADIANS, (
        "The moves that pitch back through the sphere must leave the camera standing at the near pole for the same reason. "
        f"{polar_angles=} {POLE_REACHED_RADIANS=}"
    )


def test_the_roll_lock_callback_holds_from_an_eye_off_the_lock_axis() -> None:
    """A panel seeded clear of the lock axis has a framing the roll lock is not degenerate on, and it must leave that same pitch level, upright and drawable at every pointer move of it.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_pitch(
        eye=list(OFF_AXIS_EYE),
        up=normalize_vector(vector=list(NON_AXIS_ALIGNED_LOCK_ROLL)),
    )

    assert_roll_locked_camera(records=records)


def test_the_roll_lock_callback_holds_from_an_eye_on_the_lock_axis() -> None:
    """A panel seeded looking straight down the lock axis reports a camera whose polar angle is 0, and the roll lock must hold the same pitch from there rather than aborting on the axis it is asked to hold about.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_pitch(
        eye=build_eye_along_lock_axis(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            radius=SEEDED_FRAMING_RADIUS,
        ),
        up=build_up_across_lock_axis(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
    )

    assert_roll_locked_camera(records=records)


def test_the_roll_lock_callback_holds_from_an_eye_past_the_far_pole() -> None:
    """A panel seeded looking straight up the lock axis reports a camera whose polar angle is pi, the other end of the same degeneracy, and the roll lock must hold from there too.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_pitch(
        eye=build_eye_along_lock_axis(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            radius=-SEEDED_FRAMING_RADIUS,
        ),
        up=build_up_across_lock_axis(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL),
    )

    assert_roll_locked_camera(records=records)


def test_the_roll_lock_callback_holds_from_an_already_inverted_camera() -> None:
    """A panel that comes up already hanging upside down is a framing no drag can reach, and the roll lock must put it back on the axis's own side from the first render and hold it there through the same pitch.

    Args:
        None.

    Returns:
        None.
    """
    records = run_live_pitch(
        eye=list(OFF_AXIS_EYE),
        up=build_inverted_up(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            eye=list(OFF_AXIS_EYE),
        ),
    )

    assert_roll_locked_camera(records=records)


# A camera whose eye sits on its own rotation target is the one remaining degenerate framing the callback's own normalization cannot describe, and it has no test because a Plotly gl3d panel cannot report it. `Scene.initializeGLCamera` builds the panel's camera with `zoomMin: 0.01, zoomMax: 100`, which become the view controller's radius bounds `[log(0.01), log(100)]`; the eye-to-target distance is stored as that bounded radius and `setDistance` additionally ignores any non-positive distance outright, so no drag, no wheel, and no layout-seeded camera reaches a distance of 0. Measured against the plotly.js bundle the repo's `plotly` 6.7.0 ships.
