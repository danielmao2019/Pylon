"""Dash trackball camera-control guards."""

from typing import Final

FORBIDDEN_DASH_CAMERA_CONTROL_PATTERNS: Final[tuple[str, ...]] = (
    "OrbitControls",
    ".target",
    "minPolarAngle",
    "maxPolarAngle",
    "minAzimuthAngle",
    "maxAzimuthAngle",
    "minDistance",
    "maxDistance",
    "enableRotate = false",
    "enablePan = false",
)


def create_dash_trackball_camera_controls(renderer_controls: str) -> str:
    """Create Dash renderer trackball camera controls.

    Args:
        renderer_controls: JavaScript source that owns renderer camera controls.

    Returns:
        Validated JavaScript source for trackball camera controls.
    """
    controls = create_dash_renderer_trackball_camera_controls(
        renderer_controls=renderer_controls,
    )
    assert_dash_trackball_camera_controls(controls=controls)
    return controls


def create_dash_renderer_trackball_camera_controls(renderer_controls: str) -> str:
    """Create renderer-specific Dash trackball camera controls.

    Args:
        renderer_controls: JavaScript source that owns renderer camera controls.

    Returns:
        Renderer camera-control JavaScript source.
    """
    assert isinstance(renderer_controls, str), (
        "Renderer controls must be JavaScript source. "
        "renderer_controls=%r" % renderer_controls
    )
    assert renderer_controls.strip() != "", (
        "Renderer controls source must be non-empty. "
        "renderer_controls=%r" % renderer_controls
    )
    return renderer_controls


def assert_dash_trackball_camera_controls(controls: str) -> None:
    """Assert that Dash camera controls are trackball controls.

    Args:
        controls: Renderer camera-control JavaScript source.

    Returns:
        None.
    """
    assert_dash_trackball_mouse_mapping(controls=controls)
    assert_dash_no_orbit_camera_controls(controls=controls)
    assert_dash_no_camera_pose_clamps(controls=controls)


def assert_dash_trackball_mouse_mapping(controls: str) -> None:
    """Assert that controls expose the required trackball mouse mapping.

    Args:
        controls: Renderer camera-control JavaScript source.

    Returns:
        None.
    """
    assert isinstance(controls, str), "Controls must be source text. controls=%r" % (
        controls,
    )
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


def assert_dash_no_orbit_camera_controls(controls: str) -> None:
    """Assert that controls do not use target-locked orbit semantics.

    Args:
        controls: Renderer camera-control JavaScript source.

    Returns:
        None.
    """
    assert isinstance(controls, str), "Controls must be source text. controls=%r" % (
        controls,
    )
    assert "OrbitControls" not in controls, "orbit-style camera controls are forbidden"


def assert_dash_no_camera_pose_clamps(controls: str) -> None:
    """Assert that controls do not impose camera-pose limits.

    Args:
        controls: Renderer camera-control JavaScript source.

    Returns:
        None.
    """
    assert isinstance(controls, str), "Controls must be source text. controls=%r" % (
        controls,
    )
    restricted_patterns = [
        pattern
        for pattern in FORBIDDEN_DASH_CAMERA_CONTROL_PATTERNS
        if pattern in controls
    ]
    assert not restricted_patterns, (
        "restricted camera pose controls are forbidden. restricted_patterns=%r"
        % restricted_patterns
    )
