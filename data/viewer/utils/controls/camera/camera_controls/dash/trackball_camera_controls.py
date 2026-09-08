"""Dash trackball camera-control guards."""

import math
from typing import Any, Dict, Final, Optional, Tuple, Union

FORBIDDEN_DASH_CAMERA_CONTROL_PATTERNS: Final[Tuple[str, ...]] = (
    "OrbitControls",
    ".target",
    "minPolarAngle",
    "maxPolarAngle",
    "minAzimuthAngle",
    "maxAzimuthAngle",
    "minDistance",
    "maxDistance",
    "enablePan = false",
)
FORBIDDEN_DASH_CAMERA_ROTATION_PATTERNS: Final[Tuple[str, ...]] = (
    "enableRotate = false",
)
ROLL_LOCKED_DASH_CAMERA_CONTROL_PATTERNS: Final[Tuple[str, ...]] = (
    "cameraRollLock",
    "cameraRightAxisConstraint",
)
# Plotly gl3d `layout.scene.dragmode` that carries the camera up vector along with
# the left-drag, leaving camera roll free.
PLOTLY_FREE_ROLL_DRAGMODE: Final[str] = "orbit"
# Plotly gl3d `layout.scene.dragmode` that pins the camera up vector, holding the
# camera right axis perpendicular to it and thereby locking camera roll.
PLOTLY_ROLL_LOCKED_DRAGMODE: Final[str] = "turntable"


def create_dash_trackball_camera_controls(
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Create Dash renderer trackball camera controls.

    Args:
        lock_roll: Optional axis to lock camera roll about, as an `(x, y, z)` world-space direction in the renderer's own world frame; the axis is the caller's and need not be unit length. When supplied, the controls set Plotly gl3d `layout.scene.dragmode` to `"turntable"` and pin `camera.up` to the normalized axis, so left-drag rotation holds the camera right axis perpendicular to that axis and camera roll never changes. When None, the controls set `dragmode` to `"orbit"`, which carries the up vector along with the left-drag and leaves camera roll free; `"orbit"` is not Plotly's own gl3d default of `"turntable"`, so a caller that passes nothing gets left-drag roll that the same call did not produce before this version.

    Returns:
        Plotly gl3d `layout.scene` camera configuration for trackball camera
        controls, carrying `dragmode` and, when `lock_roll` is supplied, the
        unit-length `camera.up` axis roll is held about.
    """

    def _validate_inputs() -> None:
        assert lock_roll is None or (
            isinstance(lock_roll, tuple)
            and len(lock_roll) == 3
            and all(isinstance(component, float) for component in lock_roll)
            and any(component != 0.0 for component in lock_roll)
        ), (
            "Roll lock axis must be None or a non-zero 3-tuple of floats. "
            "lock_roll=%r" % (lock_roll,)
        )

    _validate_inputs()

    controls = create_dash_renderer_trackball_camera_controls(lock_roll=lock_roll)
    assert_dash_trackball_camera_controls(controls=controls, lock_roll=lock_roll)
    return controls


def create_dash_renderer_trackball_camera_controls(
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Create renderer-specific Dash trackball camera controls.

    Plotly gl3d natively maps left-drag to rotation, right-drag to panning, and
    the mouse wheel to zoom, and draws no canvas context menu, so the renderer's
    controls are fully determined by the `layout.scene` dragmode and camera axis.

    Args:
        lock_roll: Optional axis to lock camera roll about, as an `(x, y, z)`
            world-space direction in the renderer's own world frame; the axis is
            the caller's and need not be unit length.

    Returns:
        Plotly gl3d `layout.scene` camera configuration: the free-roll dragmode
        alone when `lock_roll` is None, and the roll-locked dragmode plus the
        normalized `camera.up` axis when an axis is supplied.
    """

    def _validate_inputs() -> None:
        assert lock_roll is None or (
            isinstance(lock_roll, tuple)
            and len(lock_roll) == 3
            and all(isinstance(component, float) for component in lock_roll)
            and any(component != 0.0 for component in lock_roll)
        ), (
            "Roll lock axis must be None or a non-zero 3-tuple of floats. "
            "lock_roll=%r" % (lock_roll,)
        )

    _validate_inputs()

    if lock_roll is None:
        return {"dragmode": PLOTLY_FREE_ROLL_DRAGMODE}
    length = math.sqrt(sum(component * component for component in lock_roll))
    return {
        "dragmode": PLOTLY_ROLL_LOCKED_DRAGMODE,
        "camera": {
            "up": {
                "x": lock_roll[0] / length,
                "y": lock_roll[1] / length,
                "z": lock_roll[2] / length,
            },
        },
    }


def assert_dash_trackball_camera_controls(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that Dash camera controls are trackball controls.

    Args:
        controls: Renderer camera controls, as Plotly gl3d `layout.scene` camera
            configuration or as the three.js viewer's camera-control JavaScript
            source.
        lock_roll: Optional axis the controls lock camera roll about, as an
            `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. " "controls=%r" % (
        controls,
    )
    assert lock_roll is None or (
        isinstance(lock_roll, tuple)
        and len(lock_roll) == 3
        and all(isinstance(component, float) for component in lock_roll)
        and any(component != 0.0 for component in lock_roll)
    ), (
        "Roll lock axis must be None or a non-zero 3-tuple of floats. "
        "lock_roll=%r" % (lock_roll,)
    )
    assert_dash_trackball_mouse_mapping(controls=controls)
    assert_dash_no_orbit_camera_controls(controls=controls)
    assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
    assert_dash_roll_lock(controls=controls, lock_roll=lock_roll)


def assert_dash_trackball_mouse_mapping(controls: Union[str, Dict[str, Any]]) -> None:
    """Assert that controls expose the required trackball mouse mapping.

    Args:
        controls: Renderer camera controls, as Plotly gl3d `layout.scene` camera
            configuration or as the three.js viewer's camera-control JavaScript
            source.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. " "controls=%r" % (
        controls,
    )
    if isinstance(controls, dict):
        assert (
            "dragmode" in controls
        ), "invalid trackball camera controls. controls=%r" % (controls,)
        assert controls["dragmode"] in (
            PLOTLY_FREE_ROLL_DRAGMODE,
            PLOTLY_ROLL_LOCKED_DRAGMODE,
        ), (
            "invalid trackball camera controls. Plotly gl3d maps left-drag to "
            "rotation, right-drag to panning, and the wheel to zoom only under a "
            "rotation dragmode. dragmode=%r" % (controls["dragmode"],)
        )
        return
    required_patterns = [
        "contextmenu",
        "event.preventDefault()",
        "mousedown",
        "event.button === 2",
        "wheel",
    ]
    missing_patterns = [
        pattern for pattern in required_patterns if pattern not in controls
    ]
    assert not missing_patterns, (
        "invalid trackball camera controls. missing_patterns=%r" % missing_patterns
    )


def assert_dash_no_orbit_camera_controls(controls: Union[str, Dict[str, Any]]) -> None:
    """Assert that controls do not use target-locked orbit semantics.

    Args:
        controls: Renderer camera controls, as Plotly gl3d `layout.scene` camera
            configuration or as the three.js viewer's camera-control JavaScript
            source.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. " "controls=%r" % (
        controls,
    )
    if isinstance(controls, dict):
        assert not ("camera" in controls and "center" in controls["camera"]), (
            "orbit-style camera controls are forbidden. A pinned Plotly gl3d "
            "camera.center is a rotation target lock. controls=%r" % (controls,)
        )
        return
    assert "OrbitControls" not in controls, "orbit-style camera controls are forbidden"


def assert_dash_no_camera_pose_clamps(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that controls do not impose camera-pose limits.

    Args:
        controls: Renderer camera controls, as Plotly gl3d `layout.scene` camera
            configuration or as the three.js viewer's camera-control JavaScript
            source.
        lock_roll: Optional axis the controls lock camera roll about, as an
            `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. " "controls=%r" % (
        controls,
    )
    assert lock_roll is None or (
        isinstance(lock_roll, tuple)
        and len(lock_roll) == 3
        and all(isinstance(component, float) for component in lock_roll)
        and any(component != 0.0 for component in lock_roll)
    ), (
        "Roll lock axis must be None or a non-zero 3-tuple of floats. "
        "lock_roll=%r" % (lock_roll,)
    )
    if isinstance(controls, dict):
        assert (
            "dragmode" in controls
        ), "restricted camera pose controls are forbidden. controls=%r" % (controls,)
        if lock_roll is None:
            assert controls["dragmode"] != PLOTLY_ROLL_LOCKED_DRAGMODE, (
                "restricted camera pose controls are forbidden. The Plotly gl3d "
                "%r dragmode pins the camera up vector and thereby restricts "
                "camera roll. dragmode=%r"
                % (PLOTLY_ROLL_LOCKED_DRAGMODE, controls["dragmode"])
            )
        else:
            assert (
                controls["dragmode"] == PLOTLY_ROLL_LOCKED_DRAGMODE
            ), "roll lock must cost only the roll axis. lock_roll=%r dragmode=%r" % (
                lock_roll,
                controls["dragmode"],
            )
        return
    restricted_patterns = [
        pattern
        for pattern in FORBIDDEN_DASH_CAMERA_CONTROL_PATTERNS
        if pattern in controls
    ]
    assert not restricted_patterns, (
        "restricted camera pose controls are forbidden. restricted_patterns=%r"
        % restricted_patterns
    )
    rotation_patterns = [
        pattern
        for pattern in FORBIDDEN_DASH_CAMERA_ROTATION_PATTERNS
        if pattern in controls
    ]
    if lock_roll is None:
        assert not rotation_patterns, (
            "restricted camera pose controls are forbidden. restricted_patterns=%r"
            % rotation_patterns
        )
    else:
        assert not rotation_patterns, (
            "roll lock must cost only the roll axis. lock_roll=%r "
            "restricted_patterns=%r" % (lock_roll, rotation_patterns)
        )


def assert_dash_roll_lock(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that camera roll is held about `lock_roll` exactly when one is supplied.

    Args:
        controls: Renderer camera controls, as Plotly gl3d `layout.scene` camera
            configuration or as the three.js viewer's camera-control JavaScript
            source.
        lock_roll: Optional axis the controls lock camera roll about, as an
            `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. " "controls=%r" % (
        controls,
    )
    assert lock_roll is None or (
        isinstance(lock_roll, tuple)
        and len(lock_roll) == 3
        and all(isinstance(component, float) for component in lock_roll)
        and any(component != 0.0 for component in lock_roll)
    ), (
        "Roll lock axis must be None or a non-zero 3-tuple of floats. "
        "lock_roll=%r" % (lock_roll,)
    )
    if isinstance(controls, dict):
        if lock_roll is None:
            assert not ("camera" in controls and "up" in controls["camera"]), (
                "free trackball camera controls must leave camera roll "
                "unconstrained. controls=%r" % (controls,)
            )
            return
        assert (
            "dragmode" in controls
            and controls["dragmode"] == PLOTLY_ROLL_LOCKED_DRAGMODE
            and "camera" in controls
            and "up" in controls["camera"]
        ), (
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis. lock_roll=%r controls=%r"
            % (lock_roll, controls)
        )
        length = math.sqrt(sum(component * component for component in lock_roll))
        expected_up = {
            "x": lock_roll[0] / length,
            "y": lock_roll[1] / length,
            "z": lock_roll[2] / length,
        }
        actual_up = controls["camera"]["up"]
        assert set(actual_up) == set(expected_up) and all(
            math.isclose(
                actual_up[axis], expected_up[axis], rel_tol=1e-9, abs_tol=1e-12
            )
            for axis in expected_up
        ), (
            "roll-locked camera controls must keep the camera right axis "
            "perpendicular to the supplied axis. lock_roll=%r expected_up=%r "
            "actual_up=%r" % (lock_roll, expected_up, actual_up)
        )
        return
    if lock_roll is None:
        present_patterns = [
            pattern
            for pattern in ROLL_LOCKED_DASH_CAMERA_CONTROL_PATTERNS
            if pattern in controls
        ]
        assert not present_patterns, (
            "free trackball camera controls must leave camera roll unconstrained. "
            "present_patterns=%r" % present_patterns
        )
        return
    missing_patterns = [
        pattern
        for pattern in ROLL_LOCKED_DASH_CAMERA_CONTROL_PATTERNS
        if pattern not in controls
    ]
    assert not missing_patterns, (
        "roll-locked camera controls must keep the camera right axis perpendicular "
        "to the supplied axis. lock_roll=%r missing_patterns=%r"
        % (lock_roll, missing_patterns)
    )
