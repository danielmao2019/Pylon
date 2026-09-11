"""Dash trackball camera-control guards."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Final, Optional, Tuple, Union

import plotly.graph_objects as go
from dash import Dash, Input, dcc

FORBIDDEN_DASH_CAMERA_CONTROL_PATTERNS: Final[Tuple[str, ...]] = (
    "OrbitControls",
    ".target",
    "minAzimuthAngle",
    "maxAzimuthAngle",
    "minDistance",
    "maxDistance",
    "enablePan = false",
)
# Polar-angle limits, which a roll lock may carry and a free trackball may not: stopping the camera at the pole of the lock axis is what keeps its up vector on that axis's own side.
FORBIDDEN_DASH_CAMERA_POLAR_ANGLE_PATTERNS: Final[Tuple[str, ...]] = (
    "minPolarAngle",
    "maxPolarAngle",
)
FORBIDDEN_DASH_CAMERA_ROTATION_PATTERNS: Final[Tuple[str, ...]] = (
    "enableRotate = false",
)
# Vocabulary a roll-locked renderer source declares for holding the camera right axis perpendicular to the lock axis.
ROLL_LOCK_RIGHT_AXIS_PATTERN: Final[str] = "rollLock"
# Vocabulary a roll-locked renderer source declares for banding the polar angle off the lock axis, which is what holds the camera up vector on that axis's own side.
ROLL_LOCK_UP_SIDE_PATTERN: Final[str] = "polarAngle"
# Plotly gl3d dragmode that pins camera.up onto world +Z, and the one a scene naming no dragmode runs.
PLOTLY_POSE_CLAMPING_DRAGMODE: Final[str] = "turntable"
# Plotly gl3d dragmode whose left-drag carries camera.up with the drag, leaving roll free.
PLOTLY_FREE_ROLL_DRAGMODE: Final[str] = "orbit"
# Clientside roll-lock source `_register_dash_roll_lock_callback` inlines: one expression evaluating to a factory of the graph id and the unit-length axis.
ROLL_LOCK_CALLBACK_SCRIPT_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "roll_lock.js"
)


def apply_dash_trackball_camera_controls(
    app: Dash,
    display: dcc.Graph,
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Put one Dash Plotly gl3d display under the trackball camera controls.

    Args:
        app: Dash app serving `display`; the roll-lock clientside callback is registered on it when `lock_roll` is supplied.
        display: Plotly gl3d `dcc.Graph` already carrying its non-empty string component `id`, whose `figure` is a `go.Figure`; that figure's `layout.scene` is updated in place.
        lock_roll: Optional axis to hold camera roll about through every drag, as a non-zero `(x, y, z)` world-space direction of any length in the display's own world frame; None leaves roll free.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(app, Dash), "App must be a Dash app. type(app)=%r" % (
            type(app),
        )
        assert isinstance(
            display, dcc.Graph
        ), "Display must be a Dash `dcc.Graph`. type(display)=%r" % (type(display),)
        assert hasattr(display, "id"), (
            "Display must carry a component id, since the roll-lock callback "
            "addresses its graph by it. display_props=%r"
            % (sorted(display.to_plotly_json()["props"]),)
        )
        assert isinstance(display.id, str) and display.id != "", (
            "Display must carry a component id that is a non-empty string. "
            "display.id=%r" % (display.id,)
        )
        assert hasattr(
            display, "figure"
        ), "Display must carry a Plotly figure. display_props=%r" % (
            sorted(display.to_plotly_json()["props"]),
        )
        assert isinstance(
            display.figure, go.Figure
        ), "Display figure must be a Plotly `go.Figure`. type(display.figure)=%r" % (
            type(display.figure),
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

    _validate_inputs()

    plotly_controls: Dict[str, Any] = {"dragmode": PLOTLY_FREE_ROLL_DRAGMODE}
    if lock_roll is not None:
        length = math.sqrt(sum(component * component for component in lock_roll))
        plotly_controls["camera"] = {
            "up": {
                "x": lock_roll[0] / length,
                "y": lock_roll[1] / length,
                "z": lock_roll[2] / length,
            },
        }
    controls = create_dash_trackball_camera_controls(
        renderer_controls=plotly_controls,
        lock_roll=lock_roll,
    )
    display.figure.update_layout(scene=controls)
    if lock_roll is not None:
        _register_dash_roll_lock_callback(
            app=app,
            graph_id=display.id,
            lock_roll=lock_roll,
        )


def create_dash_trackball_camera_controls(
    renderer_controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> Union[str, Dict[str, Any]]:
    """Create Dash renderer trackball camera controls.

    Args:
        renderer_controls: A renderer's own camera-control JavaScript source, or a Plotly gl3d `layout.scene` configuration.
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction of any length; None validates the free trackball.

    Returns:
        The validated controls, exactly as `renderer_controls` arrived.
    """
    controls = create_dash_renderer_trackball_camera_controls(
        renderer_controls=renderer_controls,
    )
    assert_dash_trackball_camera_controls(controls=controls, lock_roll=lock_roll)
    return controls


def create_dash_renderer_trackball_camera_controls(
    renderer_controls: Union[str, Dict[str, Any]],
) -> Union[str, Dict[str, Any]]:
    """Create renderer-specific Dash trackball camera controls.

    Args:
        renderer_controls: A renderer's own camera-control JavaScript source, or a Plotly gl3d `layout.scene` configuration, whose left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression the renderer itself wires.

    Returns:
        The renderer controls exactly as they arrived, so a display handing over its own source renders what it rendered before.
    """

    def _validate_inputs() -> None:
        assert isinstance(renderer_controls, (str, dict)), (
            "Renderer controls must be JavaScript source or a Plotly gl3d scene "
            "configuration. renderer_controls=%r" % (renderer_controls,)
        )
        if isinstance(renderer_controls, str):
            assert renderer_controls.strip() != "", (
                "Renderer controls source must be non-empty. "
                "renderer_controls=%r" % (renderer_controls,)
            )

    _validate_inputs()

    return renderer_controls


def _register_dash_roll_lock_callback(
    app: Dash,
    graph_id: str,
    lock_roll: Tuple[float, float, float],
) -> None:
    """Register the clientside callback holding one Dash graph's camera roll about an axis at every pointer move of a drag.

    Args:
        app: Dash app the callback is registered on.
        graph_id: Component id of the `dcc.Graph` whose gl3d camera the callback holds; `dcc.Graph` renders it onto a wrapper div, inside which the callback resolves the Plotly graph div.
        lock_roll: Axis to hold camera roll about, as a non-zero `(x, y, z)` world-space direction of any length in the rendered scene's own world frame.

    Returns:
        None.
    """
    script = ROLL_LOCK_CALLBACK_SCRIPT_PATH.read_text()
    assert_dash_roll_lock(controls=script, lock_roll=lock_roll)
    length = math.sqrt(sum(component * component for component in lock_roll))
    axis = [component / length for component in lock_roll]
    app.clientside_callback(
        "(%s)(%s, %s)" % (script, json.dumps(graph_id), json.dumps(axis)),
        Input(graph_id, "relayoutData"),
    )


def assert_dash_trackball_camera_controls(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that Dash camera controls satisfy every trackball contract.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or a Plotly gl3d `layout.scene` configuration.
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """
    assert_dash_trackball_mouse_mapping(controls=controls)
    assert_dash_no_orbit_camera_controls(controls=controls)
    assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
    assert_dash_roll_lock(controls=controls, lock_roll=lock_roll)


def assert_dash_trackball_mouse_mapping(controls: Union[str, Dict[str, Any]]) -> None:
    """Assert that controls map left-drag to rotate, right-drag to pan, and wheel to zoom, with the canvas context menu suppressed.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or a Plotly gl3d `layout.scene` configuration.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. controls=%r" % (
        controls,
    )
    if isinstance(controls, dict):
        if "dragmode" in controls:
            # Plotly wires the three-button mapping and suppresses the context menu natively only under a rotation dragmode.
            assert controls["dragmode"] in (
                PLOTLY_FREE_ROLL_DRAGMODE,
                PLOTLY_POSE_CLAMPING_DRAGMODE,
            ), (
                "invalid trackball camera controls. Plotly gl3d maps left-drag to "
                "rotation, right-drag to panning, and the wheel to zoom only under "
                "a rotation dragmode. dragmode=%r" % (controls["dragmode"],)
            )
        return
    missing_mapping_patterns = [
        pattern
        for pattern in ("mousedown", "event.button === 2", "wheel")
        if pattern not in controls
    ]
    assert (
        not missing_mapping_patterns
    ), "invalid trackball camera controls. missing_patterns=%r" % (
        missing_mapping_patterns,
    )
    missing_context_menu_patterns = [
        pattern
        for pattern in ("contextmenu", "event.preventDefault()")
        if pattern not in controls
    ]
    assert (
        not missing_context_menu_patterns
    ), "context menu blocks trackball panning. missing_patterns=%r" % (
        missing_context_menu_patterns,
    )


def assert_dash_no_orbit_camera_controls(controls: Union[str, Dict[str, Any]]) -> None:
    """Assert that controls do not use target-locked orbit semantics.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or a Plotly gl3d `layout.scene` configuration.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. controls=%r" % (
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
    """Assert that controls impose no camera-pose limit beyond the polar band a roll lock costs.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or a Plotly gl3d `layout.scene` configuration.
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction; None asserts the free trackball, whose polar angle and rotation are both unrestricted.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. controls=%r" % (
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
            and controls["dragmode"] != PLOTLY_POSE_CLAMPING_DRAGMODE
        ), (
            "restricted camera pose controls are forbidden. The Plotly gl3d %r "
            "dragmode clamps the camera up vector onto world +Z, and a scene "
            "configuration naming no dragmode runs exactly that gl3d default. "
            "controls=%r" % (PLOTLY_POSE_CLAMPING_DRAGMODE, controls)
        )
        return
    restricted_patterns = [
        pattern
        for pattern in FORBIDDEN_DASH_CAMERA_CONTROL_PATTERNS
        if pattern in controls
    ]
    assert (
        not restricted_patterns
    ), "restricted camera pose controls are forbidden. restricted_patterns=%r" % (
        restricted_patterns,
    )
    polar_angle_patterns = [
        pattern
        for pattern in FORBIDDEN_DASH_CAMERA_POLAR_ANGLE_PATTERNS
        if pattern in controls
    ]
    rotation_patterns = [
        pattern
        for pattern in FORBIDDEN_DASH_CAMERA_ROTATION_PATTERNS
        if pattern in controls
    ]
    if lock_roll is None:
        assert (
            not polar_angle_patterns + rotation_patterns
        ), "restricted camera pose controls are forbidden. restricted_patterns=%r" % (
            polar_angle_patterns + rotation_patterns,
        )
    else:
        assert not rotation_patterns, (
            "roll lock must cost only the roll axis and the polar extremes. "
            "lock_roll=%r restricted_patterns=%r" % (lock_roll, rotation_patterns)
        )


def assert_dash_roll_lock(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that camera roll is held about `lock_roll` when one is supplied and left free when none is.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source (the three.js viewer's on a free display, `roll_lock.js` on a locked one) or a Plotly gl3d `layout.scene` configuration.
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """
    assert isinstance(
        controls, (str, dict)
    ), "Controls must be Plotly scene configuration or source text. controls=%r" % (
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
    if lock_roll is not None:
        if isinstance(controls, dict):
            assert (
                "dragmode" in controls
                and controls["dragmode"] == PLOTLY_FREE_ROLL_DRAGMODE
                and "camera" in controls
                and "up" in controls["camera"]
                and set(controls["camera"]["up"]) == {"x", "y", "z"}
            ), (
                "roll-locked camera controls must keep the camera right axis "
                "perpendicular to the supplied axis. lock_roll=%r controls=%r"
                % (lock_roll, controls)
            )
            length = math.sqrt(sum(component * component for component in lock_roll))
            axis = [component / length for component in lock_roll]
            up = [controls["camera"]["up"][key] for key in ("x", "y", "z")]
            up_across_axis = [
                up[1] * axis[2] - up[2] * axis[1],
                up[2] * axis[0] - up[0] * axis[2],
                up[0] * axis[1] - up[1] * axis[0],
            ]
            assert all(
                math.isclose(component, 0.0, abs_tol=1e-9)
                for component in up_across_axis
            ), (
                "roll-locked camera controls must keep the camera right axis "
                "perpendicular to the supplied axis. lock_roll=%r up=%r "
                "up_across_axis=%r" % (lock_roll, up, up_across_axis)
            )
            up_along_axis = sum(
                up_component * axis_component
                for up_component, axis_component in zip(up, axis, strict=True)
            )
            assert up_along_axis > 0.0, (
                "roll-locked camera controls must keep the camera up vector on the "
                "supplied axis's side. lock_roll=%r up=%r up_along_axis=%r"
                % (lock_roll, up, up_along_axis)
            )
        else:
            assert ROLL_LOCK_RIGHT_AXIS_PATTERN in controls, (
                "roll-locked camera controls must keep the camera right axis "
                "perpendicular to the supplied axis. lock_roll=%r missing_pattern=%r"
                % (lock_roll, ROLL_LOCK_RIGHT_AXIS_PATTERN)
            )
            assert ROLL_LOCK_UP_SIDE_PATTERN in controls, (
                "roll-locked camera controls must keep the camera up vector on the "
                "supplied axis's side. lock_roll=%r missing_pattern=%r"
                % (lock_roll, ROLL_LOCK_UP_SIDE_PATTERN)
            )
    else:
        if isinstance(controls, dict):
            assert not ("camera" in controls and "up" in controls["camera"]), (
                "free trackball camera controls must leave camera roll "
                "unconstrained. controls=%r" % (controls,)
            )
        else:
            present_patterns = [
                pattern
                for pattern in (ROLL_LOCK_RIGHT_AXIS_PATTERN, ROLL_LOCK_UP_SIDE_PATTERN)
                if pattern in controls
            ]
            assert not present_patterns, (
                "free trackball camera controls must leave camera roll "
                "unconstrained. present_patterns=%r" % (present_patterns,)
            )
