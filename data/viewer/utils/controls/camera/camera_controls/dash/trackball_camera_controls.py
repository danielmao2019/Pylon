"""Dash trackball camera controls, their guards, and the one clientside roll-lock callback."""

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, Final, Optional, Tuple, Union
from uuid import uuid4

from dash import ALL, Input, clientside_callback

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
# Plotly gl3d aspectmode that draws every axis at its data's own proportions, so a world direction keeps its angles in the scene's normalized space.
PLOTLY_DATA_PROPORTION_ASPECTMODE: Final[str] = "data"
# Type of the pattern-matching component id a roll-locked Plotly gl3d display's dcc.Graph carries, the key the roll-lock callback matches it on.
ROLL_LOCKED_GRAPH_ID_TYPE: Final[str] = "dash-roll-locked-graph"
# Text of roll_lock.js beside this module, the clientside roll-lock source this module registers: one expression evaluating to the callback over every roll-locked graph's relayoutData.
ROLL_LOCK_CALLBACK_SCRIPT: Final[str] = (
    Path(__file__).resolve().parent / "roll_lock.js"
).read_text(encoding="utf-8")


def create_dash_trackball_camera_controls(
    renderer_controls: Optional[str] = None,
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> Union[str, Dict[str, Any]]:
    """Build and validate the Dash trackball controls that every 3D Dash spatial display must use.

    Args:
        renderer_controls: A renderer's own camera-control JavaScript source, or None for a Plotly gl3d display, whose trackball is Plotly's own.
        lock_roll: Optional axis to hold camera roll about through every drag, as a non-zero `(x, y, z)` world-space direction of any length in the display's own world frame; None leaves roll free.

    Returns:
        The validated controls: `renderer_controls` exactly as it arrived when one is handed over, otherwise the Plotly gl3d controls, a dict of exactly `"scene"`, the `layout.scene` configuration the display's figure carries (the orbit dragmode, plus the data aspectmode and the unit-length lock axis as `camera.up` when `lock_roll` is supplied), and `"graph_id"`, the component id the display's `dcc.Graph` carries: None for the free trackball, or the pattern-matching id `{"type": ROLL_LOCKED_GRAPH_ID_TYPE, "index": <hex str unique per construction>, "lock_roll": <base64 of the JSON list of the unit-length axis>}` the roll-lock callback matches.
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

    controls = create_dash_renderer_trackball_camera_controls(
        renderer_controls=renderer_controls,
        lock_roll=lock_roll,
    )
    assert_dash_trackball_camera_controls(controls=controls, lock_roll=lock_roll)
    return controls


def create_dash_renderer_trackball_camera_controls(
    renderer_controls: Optional[str],
    lock_roll: Optional[Tuple[float, float, float]],
) -> Union[str, Dict[str, Any]]:
    """Construct the Dash renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.

    Args:
        renderer_controls: A renderer's own camera-control JavaScript source, which wires the mouse mapping and context-menu suppression itself, or None for a Plotly gl3d display, whose Plotly wires them.
        lock_roll: Optional axis to hold camera roll about, as a non-zero `(x, y, z)` world-space direction of any length; None leaves roll free.

    Returns:
        `renderer_controls` exactly as it arrived when one is handed over; otherwise the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).
    """

    def _validate_inputs() -> None:
        assert renderer_controls is None or (
            isinstance(renderer_controls, str) and renderer_controls.strip() != ""
        ), (
            "Renderer controls must be None or non-empty JavaScript source. "
            "renderer_controls=%r" % (renderer_controls,)
        )

    _validate_inputs()

    if renderer_controls is not None:
        # Exactly as they arrived, so a display handing over its own source renders what it rendered before lock_roll existed.
        return renderer_controls
    # The layout.scene configuration a Plotly gl3d display's figure carries and the component id its dcc.Graph carries, Plotly itself wiring left-button rotation, right-button panning, mouse-wheel zoom, and the suppressed canvas context menu.
    plotly_controls: Dict[str, Any] = {
        "scene": {"dragmode": PLOTLY_FREE_ROLL_DRAGMODE},
        "graph_id": None,
    }
    if lock_roll is not None:
        length = math.hypot(*lock_roll)
        axis = [lock_roll[0] / length, lock_roll[1] / length, lock_roll[2] / length]
        # The scene keeps world directions, so the camera.up below sits on the lock from the first frame and a data-extent change leaves the lock axis in place.
        plotly_controls["scene"]["aspectmode"] = PLOTLY_DATA_PROPORTION_ASPECTMODE
        plotly_controls["scene"]["camera"] = {
            "up": {"x": axis[0], "y": axis[1], "z": axis[2]},
        }
        # The index keeps two roll-locked graphs on one page apart; lock_roll hands the callback this graph's axis, base64 so no id value holds a "." Dash escapes in output ids.
        plotly_controls["graph_id"] = {
            "type": ROLL_LOCKED_GRAPH_ID_TYPE,
            "index": uuid4().hex,
            "lock_roll": base64.b64encode(json.dumps(axis).encode()).decode(),
        }
    return plotly_controls


def assert_dash_trackball_camera_controls(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that Dash camera controls satisfy every trackball contract.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """
    assert_dash_trackball_mouse_mapping(controls=controls)
    assert_dash_no_orbit_camera_controls(controls=controls)
    assert_dash_no_camera_pose_clamps(controls=controls, lock_roll=lock_roll)
    assert_dash_roll_lock(controls=controls, lock_roll=lock_roll)
    return


def assert_dash_trackball_mouse_mapping(controls: Union[str, Dict[str, Any]]) -> None:
    """Assert that controls map left-drag to rotate, right-drag to pan, and wheel to zoom, with the canvas context menu suppressed.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(controls, str) or (
            isinstance(controls, dict)
            and set(controls) == {"scene", "graph_id"}
            and isinstance(controls["scene"], dict)
        ), (
            "Controls must be renderer source text or Plotly gl3d controls carrying "
            "exactly a scene configuration dict and a graph id. controls=%r"
            % (controls,)
        )

    _validate_inputs()

    if isinstance(controls, dict):
        if "dragmode" in controls["scene"]:
            # Plotly wires the three-button mapping and suppresses the context menu natively only under a rotation dragmode.
            assert controls["scene"]["dragmode"] in (
                PLOTLY_FREE_ROLL_DRAGMODE,
                PLOTLY_POSE_CLAMPING_DRAGMODE,
            ), (
                "invalid trackball camera controls. Plotly gl3d maps left-drag to "
                "rotation, right-drag to panning, and the wheel to zoom only under "
                "a rotation dragmode. dragmode=%r" % (controls["scene"]["dragmode"],)
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
    return


def assert_dash_no_orbit_camera_controls(controls: Union[str, Dict[str, Any]]) -> None:
    """Assert that controls do not use target-locked orbit semantics.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(controls, str) or (
            isinstance(controls, dict)
            and set(controls) == {"scene", "graph_id"}
            and isinstance(controls["scene"], dict)
        ), (
            "Controls must be renderer source text or Plotly gl3d controls carrying "
            "exactly a scene configuration dict and a graph id. controls=%r"
            % (controls,)
        )

    _validate_inputs()

    if isinstance(controls, dict):
        assert not (
            "camera" in controls["scene"] and "center" in controls["scene"]["camera"]
        ), (
            "orbit-style camera controls are forbidden. A pinned Plotly gl3d "
            "camera.center is a rotation target lock. controls=%r" % (controls,)
        )
        return
    assert "OrbitControls" not in controls, "orbit-style camera controls are forbidden"
    return


def assert_dash_no_camera_pose_clamps(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that controls impose no camera-pose limit beyond the polar band a roll lock costs.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source or the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction; None asserts the free trackball, whose polar angle and rotation are both unrestricted.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(controls, str) or (
            isinstance(controls, dict)
            and set(controls) == {"scene", "graph_id"}
            and isinstance(controls["scene"], dict)
        ), (
            "Controls must be renderer source text or Plotly gl3d controls carrying "
            "exactly a scene configuration dict and a graph id. controls=%r"
            % (controls,)
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

    if isinstance(controls, dict):
        assert (
            "dragmode" in controls["scene"]
            and controls["scene"]["dragmode"] != PLOTLY_POSE_CLAMPING_DRAGMODE
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
    if lock_roll is not None:
        assert not rotation_patterns, (
            "roll lock must cost only the roll axis and the polar extremes. "
            "lock_roll=%r restricted_patterns=%r" % (lock_roll, rotation_patterns)
        )
    return


def assert_dash_roll_lock(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that camera roll is held about `lock_roll` when one is supplied and left free when none is.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source (the three.js viewer's on a free display, `roll_lock.js` for the roll-lock callback) or the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction; None asserts the free trackball.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(controls, str) or (
            isinstance(controls, dict)
            and set(controls) == {"scene", "graph_id"}
            and isinstance(controls["scene"], dict)
        ), (
            "Controls must be renderer source text or Plotly gl3d controls carrying "
            "exactly a scene configuration dict and a graph id. controls=%r"
            % (controls,)
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

    if lock_roll is not None:
        if isinstance(controls, dict):
            assert (
                "dragmode" in controls["scene"]
                and controls["scene"]["dragmode"] == PLOTLY_FREE_ROLL_DRAGMODE
                and "camera" in controls["scene"]
                and "up" in controls["scene"]["camera"]
                and set(controls["scene"]["camera"]["up"]) == {"x", "y", "z"}
            ), (
                "roll-locked camera controls must keep the camera right axis "
                "perpendicular to the supplied axis. lock_roll=%r controls=%r"
                % (lock_roll, controls)
            )
            length = math.hypot(*lock_roll)
            axis = [lock_roll[0] / length, lock_roll[1] / length, lock_roll[2] / length]
            up = [controls["scene"]["camera"]["up"][key] for key in ("x", "y", "z")]
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
            # A stretched scene turns the world axis away from the direction the seeded camera.up names.
            assert (
                "aspectmode" in controls["scene"]
                and controls["scene"]["aspectmode"] == PLOTLY_DATA_PROPORTION_ASPECTMODE
            ), (
                "roll-locked Plotly controls must draw the scene at its data's own "
                "proportions. lock_roll=%r scene=%r" % (lock_roll, controls["scene"])
            )
            # The roll-lock callback finds this graph, and the axis it holds the graph about, only through this id.
            assert (
                isinstance(controls["graph_id"], dict)
                and set(controls["graph_id"]) == {"type", "index", "lock_roll"}
                and controls["graph_id"]["type"] == ROLL_LOCKED_GRAPH_ID_TYPE
                and isinstance(controls["graph_id"]["index"], str)
                and isinstance(controls["graph_id"]["lock_roll"], str)
            ), (
                "roll-locked Plotly controls must carry the graph id the roll-lock "
                "callback matches, with the supplied axis. lock_roll=%r graph_id=%r"
                % (lock_roll, controls["graph_id"])
            )
            graph_axis = json.loads(base64.b64decode(controls["graph_id"]["lock_roll"]))
            assert (
                isinstance(graph_axis, list)
                and len(graph_axis) == 3
                and all(isinstance(component, float) for component in graph_axis)
                and all(
                    math.isclose(graph_component, axis_component, abs_tol=1e-9)
                    for graph_component, axis_component in zip(
                        graph_axis, axis, strict=True
                    )
                )
            ), (
                "roll-locked Plotly controls must carry the graph id the roll-lock "
                "callback matches, with the supplied axis. lock_roll=%r axis=%r "
                "graph_axis=%r" % (lock_roll, axis, graph_axis)
            )
            # The callback that id is matched by holds the lock only as far as its own source does.
            assert_dash_roll_lock(
                controls=ROLL_LOCK_CALLBACK_SCRIPT,
                lock_roll=lock_roll,
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
            assert (
                not (
                    "camera" in controls["scene"]
                    and "up" in controls["scene"]["camera"]
                )
                and controls["graph_id"] is None
            ), (
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
    return


# Module-load registration of the one roll-lock callback, ahead of every Dash app's server setup, so a roll-locked graph a callback adds after the page loaded is matched like one built into the layout.
clientside_callback(
    ROLL_LOCK_CALLBACK_SCRIPT,
    Input(
        {"type": ROLL_LOCKED_GRAPH_ID_TYPE, "index": ALL, "lock_roll": ALL},
        "relayoutData",
    ),
)
