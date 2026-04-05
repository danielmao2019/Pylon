"""Shared Three.js camera-sync helpers for Dash apps and iframe viewers."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dash import Dash
from flask import Response

MODULE_DIR = Path(__file__).resolve().parent
THREEJS_CAMERA_SYNC_HOST_TEMPLATE_PATH = MODULE_DIR / "threejs_camera_sync_host.js"
THREEJS_CAMERA_SYNC_IFRAME_TEMPLATE_PATH = MODULE_DIR / "threejs_camera_sync_iframe.js"


def register_threejs_camera_sync(
    app: Dash,
) -> str:
    """Register the parent-page Three.js camera-sync relay on one Dash app.

    Args:
        app: Dash application that will serve and load the shared relay script.

    Returns:
        Script URL path registered on the Dash app.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), (
            "Expected `app` to be a Dash instance. " f"{type(app)=}"
        )

    validate_inputs()

    script_url_path = _build_threejs_camera_sync_host_script_url()
    script_source = _build_threejs_camera_sync_host_script()
    _register_threejs_camera_sync_route(
        app=app,
        script_url_path=script_url_path,
        script_source=script_source,
    )
    _append_dash_script_url(
        app=app,
        script_url_path=script_url_path,
    )
    return script_url_path


def build_threejs_camera_sync_script(
    viewer_id: str,
    camera_sync_group: Optional[str],
) -> str:
    """Build one iframe-local Three.js camera-sync script.

    Args:
        viewer_id: Unique iframe viewer id.
        camera_sync_group: Optional browser-side camera-sync group id.

    Returns:
        JavaScript source code string, or an empty string when sync is disabled.
    """

    def validate_inputs() -> None:
        assert isinstance(viewer_id, str), (
            "Expected `viewer_id` to be a string. " f"{type(viewer_id)=}"
        )
        assert viewer_id.strip() != "", (
            "Expected `viewer_id` to be non-empty after trimming. " f"{viewer_id=}"
        )
        assert camera_sync_group is None or isinstance(camera_sync_group, str), (
            "Expected `camera_sync_group` to be `None` or a string. "
            f"{type(camera_sync_group)=}"
        )

    validate_inputs()

    normalized_viewer_id = viewer_id.strip()
    normalized_camera_sync_group = None
    if camera_sync_group is not None:
        normalized_camera_sync_group = camera_sync_group.strip()

    if normalized_camera_sync_group in (None, ""):
        return ""

    template_source = _load_threejs_camera_sync_template(
        template_path=THREEJS_CAMERA_SYNC_IFRAME_TEMPLATE_PATH,
    )
    return template_source.replace(
        "__VIEWER_ID_JSON__",
        repr(normalized_viewer_id),
    ).replace(
        "__CAMERA_SYNC_GROUP_JSON__",
        repr(normalized_camera_sync_group),
    )


def _build_threejs_camera_sync_host_script_url() -> str:
    """Build the stable served URL path for the parent relay script.

    Args:
        None.

    Returns:
        Absolute app-local URL path for the generated relay script.
    """

    return "/__shared_threejs_camera_sync/host.js"


def _build_threejs_camera_sync_host_script() -> str:
    """Build the parent-page Three.js camera-sync relay script.

    Args:
        None.

    Returns:
        JavaScript source code string for the parent-page relay.
    """

    return _load_threejs_camera_sync_template(
        template_path=THREEJS_CAMERA_SYNC_HOST_TEMPLATE_PATH,
    )


def _register_threejs_camera_sync_route(
    app: Dash,
    script_url_path: str,
    script_source: str,
) -> None:
    """Register one Flask route that serves the parent relay script.

    Args:
        app: Dash application that owns the Flask server.
        script_url_path: URL path that should serve the generated script.
        script_source: JavaScript source code to serve at that URL path.

    Returns:
        None.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), (
            "Expected `app` to be a Dash instance. " f"{type(app)=}"
        )
        assert isinstance(script_url_path, str), (
            "Expected `script_url_path` to be a string. " f"{type(script_url_path)=}"
        )
        assert script_url_path.startswith("/"), (
            "Expected `script_url_path` to start with '/'. " f"{script_url_path=}"
        )
        assert isinstance(script_source, str), (
            "Expected `script_source` to be a string. " f"{type(script_source)=}"
        )
        assert script_source != "", (
            "Expected `script_source` to be non-empty. " f"{script_source=}"
        )

    validate_inputs()

    endpoint_name = script_url_path.replace("/", "_").replace(".", "_")
    if endpoint_name in app.server.view_functions:
        return

    def serve_threejs_camera_sync_script() -> Response:
        """Serve the generated Three.js camera-sync relay script.

        Args:
            None.

        Returns:
            JavaScript response for browser loading.
        """

        return Response(script_source, mimetype="text/javascript")

    app.server.add_url_rule(
        rule=script_url_path,
        endpoint=endpoint_name,
        view_func=serve_threejs_camera_sync_script,
    )


def _append_dash_script_url(
    app: Dash,
    script_url_path: str,
) -> None:
    """Append one external script URL on the Dash app exactly once.

    Args:
        app: Dash application that should load the shared script.
        script_url_path: URL path that serves the generated script.

    Returns:
        None.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), (
            "Expected `app` to be a Dash instance. " f"{type(app)=}"
        )
        assert isinstance(script_url_path, str), (
            "Expected `script_url_path` to be a string. " f"{type(script_url_path)=}"
        )
        assert script_url_path.startswith("/"), (
            "Expected `script_url_path` to start with '/'. " f"{script_url_path=}"
        )

    validate_inputs()

    existing_external_scripts = {
        script_entry["src"] if isinstance(script_entry, dict) else script_entry
        for script_entry in app.config.external_scripts
    }
    if script_url_path in existing_external_scripts:
        return

    app.config.external_scripts.append(script_url_path)


@lru_cache(maxsize=None)
def _load_threejs_camera_sync_template(
    template_path: Path,
) -> str:
    """Load one shared Three.js camera-sync template from disk.

    Args:
        template_path: Filesystem path to one JavaScript template file.

    Returns:
        Template file contents.
    """

    def validate_inputs() -> None:
        assert isinstance(template_path, Path), (
            "Expected `template_path` to be a `Path`. " f"{type(template_path)=}"
        )
        assert template_path.is_file(), (
            "Expected `template_path` to point to an existing file. "
            f"{template_path=}"
        )
        assert template_path.suffix == ".js", (
            "Expected `template_path` to point to a JavaScript file. "
            f"{template_path=}"
        )

    validate_inputs()

    normalized_template_path = template_path.resolve()
    return normalized_template_path.read_text(encoding="utf-8")
