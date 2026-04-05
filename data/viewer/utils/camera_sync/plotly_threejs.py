"""Shared mixed Plotly/Three.js camera-sync helpers."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import List

from dash import Dash
from flask import Response

MODULE_DIR = Path(__file__).resolve().parent
PLOTLY_THREEJS_CAMERA_SYNC_TEMPLATE_PATH = MODULE_DIR / "plotly_threejs_camera_sync.js"


def register_plotly_threejs_camera_sync(
    app: Dash,
    graph_ids: List[str],
    iframe_ids: List[str],
    camera_sync_group: str,
) -> str:
    """Register one served mixed Plotly/Three.js camera-sync script on a Dash app.

    Args:
        app: Dash application that will serve and load the generated script.
        graph_ids: Ordered Plotly graph ids that should participate in synchronization.
        iframe_ids: Ordered iframe ids that should participate in synchronization.
        camera_sync_group: Shared Three.js iframe camera-sync group id.

    Returns:
        Script URL path registered on the Dash app.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), (
            "Expected `app` to be a Dash instance. " f"{type(app)=}"
        )

        _validate_plotly_threejs_camera_sync_targets(
            graph_ids=graph_ids,
            iframe_ids=iframe_ids,
            camera_sync_group=camera_sync_group,
        )

    validate_inputs()

    script_source = build_plotly_threejs_camera_sync_script(
        graph_ids=graph_ids,
        iframe_ids=iframe_ids,
        camera_sync_group=camera_sync_group,
    )
    script_url_path = _build_plotly_threejs_camera_sync_script_url(
        graph_ids=graph_ids,
        iframe_ids=iframe_ids,
        camera_sync_group=camera_sync_group,
    )
    _register_plotly_threejs_camera_sync_route(
        app=app,
        script_url_path=script_url_path,
        script_source=script_source,
    )
    _append_dash_script_url(
        app=app,
        script_url_path=script_url_path,
    )
    return script_url_path


def build_plotly_threejs_camera_sync_script(
    graph_ids: List[str],
    iframe_ids: List[str],
    camera_sync_group: str,
) -> str:
    """Build one browser script that synchronizes Plotly graphs and Three.js iframes.

    Args:
        graph_ids: Ordered Plotly graph ids that should participate in synchronization.
        iframe_ids: Ordered iframe ids that should participate in synchronization.
        camera_sync_group: Shared Three.js iframe camera-sync group id.

    Returns:
        JavaScript source code string for mixed Plotly/Three.js camera sync.
    """

    def validate_inputs() -> None:
        _validate_plotly_threejs_camera_sync_targets(
            graph_ids=graph_ids,
            iframe_ids=iframe_ids,
            camera_sync_group=camera_sync_group,
        )

    def normalize_inputs() -> tuple[List[str], List[str], str]:
        return _normalize_plotly_threejs_camera_sync_targets(
            graph_ids=graph_ids,
            iframe_ids=iframe_ids,
            camera_sync_group=camera_sync_group,
        )

    validate_inputs()

    (
        normalized_graph_ids,
        normalized_iframe_ids,
        normalized_camera_sync_group,
    ) = normalize_inputs()

    assert all(graph_id != "" for graph_id in normalized_graph_ids), (
        "Expected non-empty graph ids after trimming. " f"{normalized_graph_ids=}"
    )
    assert all(iframe_id != "" for iframe_id in normalized_iframe_ids), (
        "Expected non-empty iframe ids after trimming. " f"{normalized_iframe_ids=}"
    )

    template_source = _load_plotly_threejs_camera_sync_template(
        template_path=PLOTLY_THREEJS_CAMERA_SYNC_TEMPLATE_PATH,
    )
    script_source = (
        template_source.replace(
            "__GRAPH_IDS_JSON__",
            json.dumps(normalized_graph_ids),
        )
        .replace(
            "__IFRAME_IDS_JSON__",
            json.dumps(normalized_iframe_ids),
        )
        .replace(
            "__CAMERA_SYNC_GROUP_JSON__",
            json.dumps(normalized_camera_sync_group),
        )
    )
    return script_source


def _validate_plotly_threejs_camera_sync_targets(
    graph_ids: List[str],
    iframe_ids: List[str],
    camera_sync_group: str,
) -> None:
    """Validate one mixed Plotly/Three.js sync target configuration.

    Args:
        graph_ids: Ordered Plotly graph ids that should participate in synchronization.
        iframe_ids: Ordered iframe ids that should participate in synchronization.
        camera_sync_group: Shared Three.js iframe camera-sync group id.

    Returns:
        None.
    """

    assert isinstance(graph_ids, list), f"{type(graph_ids)=}"
    assert len(graph_ids) > 0, f"{graph_ids=}"
    assert all(isinstance(graph_id, str) for graph_id in graph_ids), f"{graph_ids=}"

    assert isinstance(iframe_ids, list), f"{type(iframe_ids)=}"
    assert len(iframe_ids) > 0, f"{iframe_ids=}"
    assert all(isinstance(iframe_id, str) for iframe_id in iframe_ids), f"{iframe_ids=}"

    assert isinstance(camera_sync_group, str), f"{type(camera_sync_group)=}"
    assert camera_sync_group.strip() != "", f"{camera_sync_group=}"


def _normalize_plotly_threejs_camera_sync_targets(
    graph_ids: List[str],
    iframe_ids: List[str],
    camera_sync_group: str,
) -> tuple[List[str], List[str], str]:
    """Normalize one mixed Plotly/Three.js sync target configuration.

    Args:
        graph_ids: Ordered Plotly graph ids that should participate in synchronization.
        iframe_ids: Ordered iframe ids that should participate in synchronization.
        camera_sync_group: Shared Three.js iframe camera-sync group id.

    Returns:
        Trimmed graph ids, trimmed iframe ids, and trimmed sync group id.
    """

    normalized_graph_ids = [graph_id.strip() for graph_id in graph_ids]
    normalized_iframe_ids = [iframe_id.strip() for iframe_id in iframe_ids]
    normalized_camera_sync_group = camera_sync_group.strip()
    return (
        normalized_graph_ids,
        normalized_iframe_ids,
        normalized_camera_sync_group,
    )


def _build_plotly_threejs_camera_sync_script_url(
    graph_ids: List[str],
    iframe_ids: List[str],
    camera_sync_group: str,
) -> str:
    """Build one stable served URL path for one mixed sync configuration.

    Args:
        graph_ids: Ordered Plotly graph ids that should participate in synchronization.
        iframe_ids: Ordered iframe ids that should participate in synchronization.
        camera_sync_group: Shared Three.js iframe camera-sync group id.

    Returns:
        Stable served script URL path.
    """

    def validate_inputs() -> None:
        _validate_plotly_threejs_camera_sync_targets(
            graph_ids=graph_ids,
            iframe_ids=iframe_ids,
            camera_sync_group=camera_sync_group,
        )

    def normalize_inputs() -> tuple[List[str], List[str], str]:
        return _normalize_plotly_threejs_camera_sync_targets(
            graph_ids=graph_ids,
            iframe_ids=iframe_ids,
            camera_sync_group=camera_sync_group,
        )

    validate_inputs()

    (
        normalized_graph_ids,
        normalized_iframe_ids,
        normalized_camera_sync_group,
    ) = normalize_inputs()

    config_signature = hashlib.sha1(
        json.dumps(
            {
                "graph_ids": normalized_graph_ids,
                "iframe_ids": normalized_iframe_ids,
                "camera_sync_group": normalized_camera_sync_group,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"/__shared_plotly_threejs_camera_sync/{config_signature}.js"


def _register_plotly_threejs_camera_sync_route(
    app: Dash,
    script_url_path: str,
    script_source: str,
) -> None:
    """Register one Flask route that serves one mixed sync script.

    Args:
        app: Dash application that owns the Flask server.
        script_url_path: Served URL path for the script.
        script_source: JavaScript source code to serve at that path.

    Returns:
        None.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), f"{type(app)=}"

        assert isinstance(script_url_path, str), f"{type(script_url_path)=}"
        assert script_url_path.startswith("/"), f"{script_url_path=}"

        assert isinstance(script_source, str), f"{type(script_source)=}"
        assert script_source != "", f"{script_source=}"

    validate_inputs()

    endpoint_name = (
        "shared_plotly_threejs_camera_sync_"
        f"{script_url_path.rsplit('/', maxsplit=1)[-1].removesuffix('.js')}"
    )
    if endpoint_name in app.server.view_functions:
        return

    def serve_plotly_threejs_camera_sync_script() -> Response:
        """Serve one mixed Plotly/Three.js sync script response.

        Args:
            None.

        Returns:
            JavaScript HTTP response.
        """

        return Response(
            response=script_source,
            mimetype="application/javascript",
        )

    app.server.add_url_rule(
        rule=script_url_path,
        endpoint=endpoint_name,
        view_func=serve_plotly_threejs_camera_sync_script,
    )


def _append_dash_script_url(
    app: Dash,
    script_url_path: str,
) -> None:
    """Append one served script URL to the Dash script resource list.

    Args:
        app: Dash application that should load the script.
        script_url_path: Served URL path for the script.

    Returns:
        None.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), f"{type(app)=}"

        assert isinstance(script_url_path, str), f"{type(script_url_path)=}"
        assert script_url_path.startswith("/"), f"{script_url_path=}"

    validate_inputs()

    existing_external_scripts = {
        script_entry["src"] if isinstance(script_entry, dict) else script_entry
        for script_entry in app.config.external_scripts
    }
    if script_url_path in existing_external_scripts:
        return
    app.config.external_scripts.append(script_url_path)


@lru_cache(maxsize=None)
def _load_plotly_threejs_camera_sync_template(
    template_path: Path,
) -> str:
    """Load one shared mixed camera-sync template from disk.

    Args:
        template_path: Filesystem path to one JavaScript template file.

    Returns:
        Template file contents.
    """

    def validate_inputs() -> None:
        assert isinstance(template_path, Path), f"{type(template_path)=}"
        assert template_path.is_file(), f"{template_path=}"
        assert template_path.suffix == ".js", f"{template_path=}"

    validate_inputs()

    return template_path.resolve().read_text(encoding="utf-8")
