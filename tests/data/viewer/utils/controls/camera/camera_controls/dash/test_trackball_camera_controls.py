"""Tests for the Dash trackball camera controls and their guards."""

import math
from typing import Dict, Tuple

import pytest

from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    assert_dash_no_camera_pose_clamps,
    assert_dash_roll_lock,
    assert_dash_trackball_camera_controls,
    create_dash_trackball_camera_controls,
)
from data.viewer.utils.displays.mesh.dash.core_mesh_display import (
    TEXTURED_MESH_VIEWER_SCRIPT_PATH,
)

# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# The same direction at a length far from 1, so a construction that forwards the
# caller's axis unnormalized cannot pass.
NON_UNIT_LOCK_ROLL = (3.0, 9.0, -2.0)


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
        axis on every drag step.
    """
    return build_free_trackball_renderer_controls() + """
    container.dataset.cameraRollLock = JSON.stringify(rollLockAxis);
    container.dataset.cameraRightAxisConstraint = "perpendicular-to-roll-lock-axis";
    cameraRightAxis.crossVectors(viewDirection, rollLockAxis).normalize();
    camera.up.crossVectors(cameraRightAxis, viewDirection).normalize();
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


def test_no_axis_means_free_trackball() -> None:
    """A caller that names no lock_roll gets no camera configuration at all, identical to an explicit lock_roll=None construction."""
    defaulted_controls = create_dash_trackball_camera_controls()
    explicit_controls = create_dash_trackball_camera_controls(lock_roll=None)

    assert defaulted_controls == explicit_controls, (
        "Naming no lock_roll must carry the same rotation wiring as an explicit "
        "lock_roll=None construction. "
        f"{defaulted_controls=} {explicit_controls=}"
    )
    assert defaulted_controls == {}, (
        "Naming no roll-lock axis must add no camera configuration, so the display "
        "renders the camera it rendered before this argument existed. "
        f"{defaulted_controls=}"
    )


def test_a_supplied_axis_is_held_through_a_drag() -> None:
    """Constructing with a lock_roll pins the Plotly camera up vector to that supplied axis."""
    controls = create_dash_trackball_camera_controls(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
    )

    assert controls["dragmode"] == "turntable", (
        "Roll-locked controls must select the Plotly gl3d dragmode that pins the "
        f"camera up vector. {controls=}"
    )
    assert controls["camera"]["up"] == expected_camera_up(NON_AXIS_ALIGNED_LOCK_ROLL), (
        "Roll-locked controls must pin the camera up vector to the caller's axis. "
        f"{controls=} {NON_AXIS_ALIGNED_LOCK_ROLL=}"
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


def test_roll_locked_source_holds_the_camera_right_axis() -> None:
    """Renderer source that re-derives the camera right axis passes the roll-locked contract and fails the free-trackball one."""
    controls = build_roll_locked_renderer_controls()

    assert_dash_roll_lock(controls=controls, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)
    with pytest.raises(
        AssertionError,
        match="free trackball camera controls must leave camera roll unconstrained",
    ):
        assert_dash_roll_lock(controls=controls, lock_roll=None)
