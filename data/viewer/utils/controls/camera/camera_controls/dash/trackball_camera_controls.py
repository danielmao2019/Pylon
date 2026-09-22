"""Dash trackball camera controls, their guards, and the one clientside roll-lock callback."""

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from uuid import uuid4

from dash import ALL, Input, clientside_callback

# Plotly gl3d dragmode that pins camera.up onto world +Z, and the one a scene naming no dragmode runs.
PLOTLY_POSE_CLAMPING_DRAGMODE = "turntable"
# Plotly gl3d dragmode whose left-drag carries camera.up with the drag, leaving roll free.
PLOTLY_FREE_ROLL_DRAGMODE = "orbit"
# Plotly gl3d aspectmode that draws every axis at its data's own proportions, so a world direction keeps its angles in the scene's normalized space.
PLOTLY_DATA_PROPORTION_ASPECTMODE = "data"
# Type of the pattern-matching component id a roll-locked Plotly gl3d display's dcc.Graph carries, the key the roll-lock callback matches it on.
ROLL_LOCKED_GRAPH_ID_TYPE = "dash-roll-locked-graph"
# Text of roll_lock.js beside this module, the clientside roll-lock source this module registers: one expression evaluating to the callback over every roll-locked graph's relayoutData.
ROLL_LOCK_CALLBACK_SCRIPT = (
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
        # Plotly wires the three-button mapping and suppresses the context menu natively only under a rotation dragmode.
        assert "dragmode" not in controls["scene"] or controls["scene"]["dragmode"] in (
            PLOTLY_FREE_ROLL_DRAGMODE,
            PLOTLY_POSE_CLAMPING_DRAGMODE,
        ), (
            "invalid trackball camera controls. Plotly gl3d maps left-drag to "
            "rotation, right-drag to panning, and the wheel to zoom only under "
            "a rotation dragmode. scene=%r" % (controls["scene"],)
        )
        return
    assert (
        "mousedown" in controls
        and "event.button === 2" in controls
        and "wheel" in controls
    ), (
        "invalid trackball camera controls. A renderer source wires the mapping "
        "through 'mousedown', 'event.button === 2', and 'wheel'. present=%r"
        % (
            {
                "mousedown": "mousedown" in controls,
                "event.button === 2": "event.button === 2" in controls,
                "wheel": "wheel" in controls,
            },
        )
    )
    assert "contextmenu" in controls and "event.preventDefault()" in controls, (
        "context menu blocks trackball panning. A renderer source suppresses it "
        "through 'contextmenu' and 'event.preventDefault()'. present=%r"
        % (
            {
                "contextmenu": "contextmenu" in controls,
                "event.preventDefault()": "event.preventDefault()" in controls,
            },
        )
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

    assert (isinstance(controls, str) and "OrbitControls" not in controls) or (
        isinstance(controls, dict)
        and (
            "camera" not in controls["scene"]
            or "center" not in controls["scene"]["camera"]
        )
    ), (
        "orbit-style camera controls are forbidden. OrbitControls in a renderer "
        "source and a pinned Plotly gl3d camera.center are both rotation target "
        "locks. controls=%.500r" % (controls,)
    )
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

    assert isinstance(controls, str) or (
        "dragmode" in controls["scene"]
        and controls["scene"]["dragmode"] != PLOTLY_POSE_CLAMPING_DRAGMODE
    ), (
        "restricted camera pose controls are forbidden. The Plotly gl3d %r "
        "dragmode clamps the camera up vector onto world +Z, and a scene "
        "configuration naming no dragmode runs exactly that gl3d default. "
        "controls=%r" % (PLOTLY_POSE_CLAMPING_DRAGMODE, controls)
    )
    assert isinstance(controls, dict) or (
        "OrbitControls" not in controls
        and ".target" not in controls
        and "minAzimuthAngle" not in controls
        and "maxAzimuthAngle" not in controls
        and "minDistance" not in controls
        and "maxDistance" not in controls
        and "enablePan = false" not in controls
    ), (
        "restricted camera pose controls are forbidden. A renderer source locks "
        "no target and bounds no azimuth angle, distance, or pan. present=%r"
        % (
            {
                "OrbitControls": "OrbitControls" in controls,
                ".target": ".target" in controls,
                "minAzimuthAngle": "minAzimuthAngle" in controls,
                "maxAzimuthAngle": "maxAzimuthAngle" in controls,
                "minDistance": "minDistance" in controls,
                "maxDistance": "maxDistance" in controls,
                "enablePan = false": "enablePan = false" in controls,
            },
        )
    )
    # Polar-angle limits are what a roll lock may carry and a free trackball may not: stopping the camera at the pole of the lock axis is what keeps its up vector on that axis's own side.
    assert (
        lock_roll is not None
        or isinstance(controls, dict)
        or (
            "minPolarAngle" not in controls
            and "maxPolarAngle" not in controls
            and "enableRotate = false" not in controls
        )
    ), (
        "restricted camera pose controls are forbidden. A free trackball's "
        "renderer source bounds no polar angle and keeps rotation enabled. "
        "present=%r"
        % (
            {
                "minPolarAngle": "minPolarAngle" in controls,
                "maxPolarAngle": "maxPolarAngle" in controls,
                "enableRotate = false": "enableRotate = false" in controls,
            },
        )
    )
    assert (
        lock_roll is None
        or isinstance(controls, dict)
        or "enableRotate = false" not in controls
    ), (
        "roll lock must cost only the roll axis and the polar extremes. A "
        "roll-locked renderer source keeps rotation enabled, yet it contains "
        "'enableRotate = false'. lock_roll=%r" % (lock_roll,)
    )
    return


def assert_dash_roll_lock(
    controls: Union[str, Dict[str, Any]],
    lock_roll: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Assert that camera roll is held about `lock_roll` when one is supplied and left free when none is.

    Args:
        controls: Renderer camera controls, as a renderer's camera-control JavaScript source (the three.js viewer's on a free display, `roll_lock.js` for the roll-lock callback) or the Plotly gl3d controls, a dict of exactly `"scene"` (the `layout.scene` configuration dict) and `"graph_id"` (None, or the roll-locked pattern-matching component id dict).
        lock_roll: Optional axis the controls hold camera roll about, as a non-zero `(x, y, z)` world-space direction of any length, checked at unit length; None asserts the free trackball.

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

    def _normalize_inputs() -> Optional[Tuple[float, float, float]]:
        if lock_roll is None:
            return None
        length = math.hypot(*lock_roll)
        return (lock_roll[0] / length, lock_roll[1] / length, lock_roll[2] / length)

    lock_roll = _normalize_inputs()

    # A roll-locked renderer source declares its hold on the camera right axis as "rollLock"; Plotly controls hold it by pinning camera.up along the axis under the orbit dragmode.
    assert (
        lock_roll is None
        or (isinstance(controls, str) and "rollLock" in controls)
        or (
            isinstance(controls, dict)
            and "dragmode" in controls["scene"]
            and controls["scene"]["dragmode"] == PLOTLY_FREE_ROLL_DRAGMODE
            and "camera" in controls["scene"]
            and "up" in controls["scene"]["camera"]
            and set(controls["scene"]["camera"]["up"]) == {"x", "y", "z"}
            and abs(
                controls["scene"]["camera"]["up"]["y"] * lock_roll[2]
                - controls["scene"]["camera"]["up"]["z"] * lock_roll[1]
            )
            <= 1e-9
            and abs(
                controls["scene"]["camera"]["up"]["z"] * lock_roll[0]
                - controls["scene"]["camera"]["up"]["x"] * lock_roll[2]
            )
            <= 1e-9
            and abs(
                controls["scene"]["camera"]["up"]["x"] * lock_roll[1]
                - controls["scene"]["camera"]["up"]["y"] * lock_roll[0]
            )
            <= 1e-9
        )
    ), (
        "roll-locked camera controls must keep the camera right axis "
        "perpendicular to the supplied axis. A renderer source declares "
        "'rollLock'; Plotly controls run the %r dragmode with camera.up along the "
        "axis. lock_roll=%r controls=%.500r"
        % (PLOTLY_FREE_ROLL_DRAGMODE, lock_roll, controls)
    )
    # A roll-locked renderer source declares its polar band off the lock axis as "polarAngle"; Plotly controls hold the side through camera.up, which the assertion above pinned along the axis.
    assert (
        lock_roll is None
        or (isinstance(controls, str) and "polarAngle" in controls)
        or (
            isinstance(controls, dict)
            and controls["scene"]["camera"]["up"]["x"] * lock_roll[0]
            + controls["scene"]["camera"]["up"]["y"] * lock_roll[1]
            + controls["scene"]["camera"]["up"]["z"] * lock_roll[2]
            > 0.0
        )
    ), (
        "roll-locked camera controls must keep the camera up vector on the "
        "supplied axis's side. A renderer source declares 'polarAngle'; Plotly "
        "controls point camera.up along the axis. lock_roll=%r controls=%.500r"
        % (lock_roll, controls)
    )
    if lock_roll is not None and isinstance(controls, dict):
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
            and isinstance(graph_axis[0], float)
            and isinstance(graph_axis[1], float)
            and isinstance(graph_axis[2], float)
            and abs(graph_axis[0] - lock_roll[0]) <= 1e-9
            and abs(graph_axis[1] - lock_roll[1]) <= 1e-9
            and abs(graph_axis[2] - lock_roll[2]) <= 1e-9
        ), (
            "roll-locked Plotly controls must carry the graph id the roll-lock "
            "callback matches, with the supplied axis. lock_roll=%r graph_axis=%r"
            % (lock_roll, graph_axis)
        )
        # The callback that id is matched by holds the lock only as far as its own source does.
        assert_dash_roll_lock(controls=ROLL_LOCK_CALLBACK_SCRIPT, lock_roll=lock_roll)
        return
    assert (
        lock_roll is not None
        or (
            isinstance(controls, dict)
            and (
                "camera" not in controls["scene"]
                or "up" not in controls["scene"]["camera"]
            )
            and controls["graph_id"] is None
        )
        or (
            isinstance(controls, str)
            and "rollLock" not in controls
            and "polarAngle" not in controls
        )
    ), (
        "free trackball camera controls must leave camera roll unconstrained. "
        "Plotly controls pin no camera.up and carry no graph id; a renderer source "
        "declares neither 'rollLock' nor 'polarAngle'. controls=%.500r" % (controls,)
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
