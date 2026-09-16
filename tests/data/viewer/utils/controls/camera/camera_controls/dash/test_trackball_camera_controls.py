"""Tests for the Dash trackball camera controls, the one roll-lock callback, and their roll-lock guards."""

import base64
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest
from dash import ALL, Input

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCK_CALLBACK_SCRIPT,
    ROLL_LOCKED_GRAPH_ID_TYPE,
    assert_dash_no_camera_pose_clamps,
    assert_dash_roll_lock,
    create_dash_trackball_camera_controls,
)
from data.viewer.utils.displays.mesh.dash.core_mesh_display import (
    TEXTURED_MESH_VIEWER_SCRIPT_PATH,
)

# Repository root, the working directory the registration probe imports the module from.
REPO_ROOT = Path(__file__).resolve().parents[8]
# Deliberately non-axis-aligned, so nothing passes by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# The same direction at ten times the length, so an axis forwarded unnormalized cannot pass.
NON_UNIT_LOCK_ROLL = (3.0, 9.0, -2.0)
# The unit-length direction both axes above name.
NORMALIZED_LOCK_ROLL = tuple(
    component / math.sqrt(sum(value * value for value in NON_AXIS_ALIGNED_LOCK_ROLL))
    for component in NON_AXIS_ALIGNED_LOCK_ROLL
)
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
# Script a fresh interpreter runs to report the callbacks importing the module registers globally on the roll-locked graph pattern, before any Dash app's server setup moves them into that app and clears the global lists.
REGISTRATION_PROBE_SCRIPT = """
import json

import dash._callback

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCKED_GRAPH_ID_TYPE,
)

print(
    json.dumps(
        {
            "callbacks": [
                callback
                for callback in dash._callback.GLOBAL_CALLBACK_LIST
                if any(
                    ROLL_LOCKED_GRAPH_ID_TYPE in callback_input["id"]
                    for callback_input in callback["inputs"]
                )
            ],
            "inline_scripts": dash._callback.GLOBAL_INLINE_SCRIPTS,
        }
    )
)
"""


def test_no_axis_builds_the_free_roll_controls() -> None:
    """Plotly controls built with no lock_roll run the free-roll orbit dragmode, pin no camera.up, and carry no graph id, so the roll-lock callback matches nothing of theirs.

    Args:
        None.

    Returns:
        None.
    """
    controls = create_dash_trackball_camera_controls()

    assert controls == {"scene": {"dragmode": "orbit"}, "graph_id": None}, (
        "Expected the free trackball to run the orbit dragmode, pin no camera, "
        f"and carry no graph id. {controls=}"
    )
    return


def test_a_supplied_axis_seeds_the_normalized_axis_as_camera_up() -> None:
    """Plotly controls built with a lock_roll carry that axis, normalized, as their scene's camera.up under the free-roll orbit dragmode.

    Args:
        None.

    Returns:
        None.
    """
    controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )

    assert controls["scene"]["dragmode"] == "orbit", (
        "Expected the roll-locked controls to run the orbit dragmode. " f"{controls=}"
    )
    assert controls["scene"]["aspectmode"] == "data", (
        "Expected the roll-locked controls to draw the scene at its data's own "
        f"proportions. {controls=}"
    )
    up = controls["scene"]["camera"]["up"]
    assert up == pytest.approx(
        {
            "x": NORMALIZED_LOCK_ROLL[0],
            "y": NORMALIZED_LOCK_ROLL[1],
            "z": NORMALIZED_LOCK_ROLL[2],
        },
        abs=1e-12,
    ), (
        "Expected camera.up to be the normalized lock_roll. "
        f"{up=} {NORMALIZED_LOCK_ROLL=}"
    )
    return


def test_a_non_unit_axis_is_normalized() -> None:
    """The caller's axis need not be unit length, so the same direction at any length pins the same camera up vector and hands the callback the same axis.

    Args:
        None.

    Returns:
        None.
    """
    unit_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )
    scaled_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_UNIT_LOCK_ROLL
    )

    unit_up = unit_controls["scene"]["camera"]["up"]
    scaled_up = scaled_controls["scene"]["camera"]["up"]
    assert scaled_up == pytest.approx(unit_up, abs=1e-12), (
        "Expected one direction at two lengths to pin one camera up vector. "
        f"{unit_up=} {scaled_up=}"
    )
    assert math.isclose(
        math.hypot(scaled_up["x"], scaled_up["y"], scaled_up["z"]), 1.0
    ), ("Expected the pinned camera up vector to be unit length. " f"{scaled_up=}")
    unit_graph_axis = json.loads(
        base64.b64decode(unit_controls["graph_id"]["lock_roll"])
    )
    scaled_graph_axis = json.loads(
        base64.b64decode(scaled_controls["graph_id"]["lock_roll"])
    )
    assert scaled_graph_axis == pytest.approx(unit_graph_axis, abs=1e-12), (
        "Expected one direction at two lengths to hand the callback one axis. "
        f"{unit_graph_axis=} {scaled_graph_axis=}"
    )
    return


def test_a_supplied_axis_carries_the_roll_locked_graph_id() -> None:
    """Plotly controls built with a lock_roll carry a ROLL_LOCKED_GRAPH_ID_TYPE component id handing the callback the normalized axis, its index unique per construction so two roll-locked graphs on one page never share an id.

    Args:
        None.

    Returns:
        None.
    """
    first_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )
    second_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )

    assert first_controls["graph_id"]["type"] == ROLL_LOCKED_GRAPH_ID_TYPE, (
        "Expected the roll-locked graph id to carry the type the callback matches. "
        f"{first_controls['graph_id']=} {ROLL_LOCKED_GRAPH_ID_TYPE=}"
    )
    graph_axis = json.loads(base64.b64decode(first_controls["graph_id"]["lock_roll"]))
    assert graph_axis == pytest.approx(NORMALIZED_LOCK_ROLL, abs=1e-12), (
        "Expected the roll-locked graph id to hand the callback the normalized "
        f"lock_roll. {graph_axis=} {NORMALIZED_LOCK_ROLL=}"
    )
    graph_id_text = "".join(first_controls["graph_id"].values())
    assert "." not in graph_id_text, (
        "Expected no roll-locked graph id value to hold a '.', which Dash escapes "
        f"in output ids. {first_controls['graph_id']=}"
    )
    assert (
        first_controls["graph_id"]["index"] != second_controls["graph_id"]["index"]
    ), (
        "Expected two constructions with one lock_roll to carry different graph id "
        f"indices. {first_controls['graph_id']=} {second_controls['graph_id']=}"
    )
    return


def test_a_supplied_axis_pins_the_data_aspect() -> None:
    """Plotly controls built with a lock_roll draw the scene at its data's own proportions, so the world axis keeps its direction in the scene's space, while free controls leave the aspect to Plotly.

    Args:
        None.

    Returns:
        None.
    """
    roll_locked_controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )
    free_controls = create_dash_trackball_camera_controls()

    assert roll_locked_controls["scene"]["aspectmode"] == "data", (
        "Expected the roll-locked scene to draw at its data's own proportions. "
        f"{roll_locked_controls['scene']=}"
    )
    assert "aspectmode" not in free_controls["scene"], (
        "Expected the free scene to leave the aspect to Plotly. "
        f"{free_controls['scene']=}"
    )
    return


def test_the_roll_lock_callback_is_registered_once_on_the_roll_locked_graph_pattern() -> (
    None
):
    """Importing the module registers exactly one clientside callback, matching every roll-locked graph's relayoutData by pattern and running the shipped roll_lock.js source, so no display registers a callback of its own.

    The registration is read in a fresh interpreter, since the first Dash app in a process to set up its server moves every global callback into itself and clears the global lists.

    Args:
        None.

    Returns:
        None.
    """
    completed = subprocess.run(
        args=[sys.executable, "-c", REGISTRATION_PROBE_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "Expected the registration probe to import the module cleanly. "
        f"{completed.returncode=} {completed.stderr[-2000:]=}"
    )
    probe = json.loads(completed.stdout)

    assert len(probe["callbacks"]) == 1, (
        "Expected exactly one global callback on the roll-locked graph pattern. "
        f"{probe['callbacks']=}"
    )
    callback = probe["callbacks"][0]
    expected_input = Input(
        {"type": ROLL_LOCKED_GRAPH_ID_TYPE, "index": ALL, "lock_roll": ALL},
        "relayoutData",
    ).to_dict()
    assert callback["inputs"] == [expected_input], (
        "Expected the callback's one input to be every roll-locked graph's "
        f"relayoutData. {callback['inputs']=} {expected_input=}"
    )
    registering_scripts = [
        script
        for script in probe["inline_scripts"]
        if callback["clientside_function"]["function_name"] in script
    ]
    assert (
        len(registering_scripts) == 1
        and ROLL_LOCK_CALLBACK_SCRIPT in registering_scripts[0]
    ), (
        "Expected the callback's one inline source to be roll_lock.js. "
        f"{len(registering_scripts)=} {callback['clientside_function']=}"
    )
    return


def test_a_zero_axis_is_rejected() -> None:
    """A zero-length lock_roll names no direction, so it is rejected rather than normalized into a NaN camera.up.

    Args:
        None.

    Returns:
        None.
    """
    with pytest.raises(AssertionError, match="non-zero 3-tuple of floats"):
        create_dash_trackball_camera_controls(lock_roll=(0.0, 0.0, 0.0))
    return


def test_roll_locked_controls_keep_every_other_degree_of_freedom_free() -> None:
    """Roll lock constrains roll alone, so roll-locked Plotly controls still pass the mouse-mapping, no-orbit, and no-pose-clamp contracts.

    Args:
        None.

    Returns:
        None.
    """
    # The factory runs every trackball contract on the controls it builds, so returning without raising is the assertion.
    create_dash_trackball_camera_controls(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
    return


def test_the_threejs_viewer_source_passes_the_trackball_contract() -> None:
    """The shipped three.js mesh viewer source, handed over the way the mesh display hands it, satisfies every trackball contract and comes back unchanged, so the display's guard keeps guarding it.

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
    return


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
    return


def test_the_roll_lock_source_holds_the_camera_right_axis_and_up_vector() -> None:
    """The shipped roll_lock.js source passes the roll-locked contract and fails the free-trackball one.

    Args:
        None.

    Returns:
        None.
    """
    assert_dash_roll_lock(
        controls=ROLL_LOCK_CALLBACK_SCRIPT, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )
    with pytest.raises(
        AssertionError,
        match="free trackball camera controls must leave camera roll unconstrained",
    ):
        assert_dash_roll_lock(controls=ROLL_LOCK_CALLBACK_SCRIPT)
    return


def test_assert_dash_roll_lock_rejects_an_ignored_flag() -> None:
    """Supplied-lock_roll Plotly controls that pin no camera.up are rejected, so the flag cannot be silently dropped.

    Args:
        None.

    Returns:
        None.
    """
    controls = {"scene": {"dragmode": "orbit"}, "graph_id": None}

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
    return


def test_assert_dash_roll_lock_rejects_a_mismatched_axis() -> None:
    """Plotly controls pinned to a different axis than the caller supplied are rejected, so the caller's axis cannot be swapped for another.

    Args:
        None.

    Returns:
        None.
    """
    controls = {
        "scene": {
            "dragmode": "orbit",
            "aspectmode": "data",
            "camera": {"up": {"x": 0.0, "y": 0.0, "z": 1.0}},
        },
        "graph_id": None,
    }

    with pytest.raises(
        AssertionError,
        match=(
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis"
        ),
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
    return


def test_assert_dash_roll_lock_rejects_a_graph_id_the_callback_does_not_match() -> None:
    """Roll-locked Plotly controls whose graph id is missing, of another type, or carrying another axis are rejected, so the roll-lock callback can neither miss the graph nor hold it about the wrong axis.

    Args:
        None.

    Returns:
        None.
    """
    controls = {
        "scene": {
            "dragmode": "orbit",
            "aspectmode": "data",
            "camera": {
                "up": {
                    "x": NORMALIZED_LOCK_ROLL[0],
                    "y": NORMALIZED_LOCK_ROLL[1],
                    "z": NORMALIZED_LOCK_ROLL[2],
                },
            },
        },
        "graph_id": {
            "type": ROLL_LOCKED_GRAPH_ID_TYPE,
            "index": "0",
            "lock_roll": base64.b64encode(
                json.dumps(NORMALIZED_LOCK_ROLL).encode()
            ).decode(),
        },
    }
    variants = [
        {"scene": controls["scene"], "graph_id": None},
        {
            "scene": controls["scene"],
            "graph_id": {**controls["graph_id"], "type": "dash-other-graph"},
        },
        {
            "scene": controls["scene"],
            "graph_id": {
                **controls["graph_id"],
                "lock_roll": base64.b64encode(
                    json.dumps([0.0, 0.0, 1.0]).encode()
                ).decode(),
            },
        },
    ]

    for variant in variants:
        with pytest.raises(
            AssertionError,
            match=(
                "roll-locked Plotly controls must carry the graph id the roll-lock "
                "callback matches, with the supplied axis"
            ),
        ):
            assert_dash_roll_lock(
                controls=variant, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
            )
    return


def test_assert_dash_roll_lock_rejects_a_scene_not_at_data_proportions() -> None:
    """Roll-locked Plotly controls whose scene leaves the aspect to Plotly are rejected, since a stretched scene turns the world axis away from the direction the seeded camera.up names.

    Args:
        None.

    Returns:
        None.
    """
    controls = {
        "scene": {
            "dragmode": "orbit",
            "aspectmode": "data",
            "camera": {
                "up": {
                    "x": NORMALIZED_LOCK_ROLL[0],
                    "y": NORMALIZED_LOCK_ROLL[1],
                    "z": NORMALIZED_LOCK_ROLL[2],
                },
            },
        },
        "graph_id": {
            "type": ROLL_LOCKED_GRAPH_ID_TYPE,
            "index": "0",
            "lock_roll": base64.b64encode(
                json.dumps(NORMALIZED_LOCK_ROLL).encode()
            ).decode(),
        },
    }
    variants = [
        {
            "scene": {"dragmode": "orbit", "camera": controls["scene"]["camera"]},
            "graph_id": controls["graph_id"],
        },
        {
            "scene": {**controls["scene"], "aspectmode": "cube"},
            "graph_id": controls["graph_id"],
        },
    ]

    for variant in variants:
        with pytest.raises(
            AssertionError,
            match=(
                "roll-locked Plotly controls must draw the scene at its data's own "
                "proportions"
            ),
        ):
            assert_dash_roll_lock(
                controls=variant, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
            )
    return


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
    return


def test_assert_dash_roll_lock_rejects_an_unrequested_lock() -> None:
    """lock_roll=None Plotly controls that nonetheless pin camera.up or carry a roll-locked graph id are rejected, so the default construction cannot quietly become roll-locked.

    Args:
        None.

    Returns:
        None.
    """
    unrequested_locks = [
        {
            "scene": {
                "dragmode": "orbit",
                "camera": {"up": {"x": 0.0, "y": 0.0, "z": 1.0}},
            },
            "graph_id": None,
        },
        {
            "scene": {"dragmode": "orbit"},
            "graph_id": {
                "type": ROLL_LOCKED_GRAPH_ID_TYPE,
                "index": "0",
                "lock_roll": base64.b64encode(
                    json.dumps(NORMALIZED_LOCK_ROLL).encode()
                ).decode(),
            },
        },
    ]

    for controls in unrequested_locks:
        with pytest.raises(
            AssertionError,
            match="free trackball camera controls must leave camera roll unconstrained",
        ):
            assert_dash_roll_lock(controls=controls, lock_roll=None)
    return


@pytest.mark.parametrize("lock_roll", [None, NON_AXIS_ALIGNED_LOCK_ROLL])
def test_assert_dash_no_camera_pose_clamps_rejects_the_pose_clamping_dragmode(
    lock_roll: Optional[Tuple[float, float, float]],
) -> None:
    """The turntable dragmode pins camera.up onto world +Z, so it is rejected as a pose clamp whether or not an axis is supplied.

    Args:
        lock_roll: Axis supplied alongside the controls, or None for the free trackball.

    Returns:
        None.
    """
    controls = {"scene": {"dragmode": "turntable"}, "graph_id": None}

    with pytest.raises(AssertionError, match="restricted camera pose controls"):
        assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
    return


@pytest.mark.parametrize("lock_roll", [None, NON_AXIS_ALIGNED_LOCK_ROLL])
def test_assert_dash_no_camera_pose_clamps_rejects_an_omitted_dragmode(
    lock_roll: Optional[Tuple[float, float, float]],
) -> None:
    """Plotly controls whose scene names no dragmode run Plotly's turntable default, so they are rejected the same way.

    Args:
        lock_roll: Axis supplied alongside the controls, or None for the free trackball.

    Returns:
        None.
    """
    controls = {
        "scene": {"camera": {"eye": {"x": 1.25, "y": 1.25, "z": 1.25}}},
        "graph_id": None,
    }

    with pytest.raises(AssertionError, match="restricted camera pose controls"):
        assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
    return
