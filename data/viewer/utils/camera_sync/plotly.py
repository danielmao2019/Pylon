"""Shared Plotly camera-sync helpers for Dash apps."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dash import Dash
from flask import Response

MODULE_DIR = Path(__file__).resolve().parent
PLOTLY_CAMERA_SYNC_TEMPLATE_PATH = MODULE_DIR / "plotly_camera_sync.js"


def register_plotly_camera_sync(
    app: Dash,
    graph_ids: Optional[List[str]] = None,
    graph_id_type: Optional[str] = None,
    camera_store_id: Optional[str] = None,
) -> str:
    """Register shared Plotly camera sync on one Dash app.

    Args:
        app: Dash application that will serve and load the generated script.
        graph_ids: Optional ordered Dash ids for Plotly 3D graph components to
            synchronize.
        graph_id_type: Optional Dash pattern-id `type` for synchronized Plotly
            graph components.
        camera_store_id: Optional Dash `dcc.Store` id that should receive the
            latest synchronized camera state.

    Returns:
        Script URL path registered on the Dash app.
    """

    def validate_inputs() -> None:
        assert isinstance(app, Dash), (
            "Expected `app` to be a Dash instance. " f"{type(app)=}"
        )
        assert graph_ids is None or isinstance(graph_ids, list), (
            "Expected `graph_ids` to be `None` or a list. " f"{type(graph_ids)=}"
        )
        if graph_ids is not None:
            assert len(graph_ids) >= 2, (
                "Expected at least two graph ids for synchronization. " f"{graph_ids=}"
            )
            for graph_id in graph_ids:
                assert isinstance(graph_id, str), (
                    "Expected each graph id to be a string. " f"{type(graph_id)=}"
                )
                assert graph_id.strip() != "", (
                    "Expected each graph id to be non-empty after trimming. "
                    f"{graph_ids=}"
                )
        assert graph_id_type is None or isinstance(graph_id_type, str), (
            "Expected `graph_id_type` to be `None` or a string. "
            f"{type(graph_id_type)=}"
        )
        if graph_id_type is not None:
            assert graph_id_type.strip() != "", (
                "Expected `graph_id_type` to be non-empty after trimming. "
                f"{graph_id_type=}"
            )
        assert camera_store_id is None or isinstance(camera_store_id, str), (
            "Expected `camera_store_id` to be `None` or a string. "
            f"{type(camera_store_id)=}"
        )
        if camera_store_id is not None:
            assert camera_store_id.strip() != "", (
                "Expected `camera_store_id` to be non-empty after trimming. "
                f"{camera_store_id=}"
            )

    validate_inputs()

    assert (graph_ids is None) != (graph_id_type is None), (
        "Expected exactly one of `graph_ids` or `graph_id_type`. "
        f"{graph_ids=}, {graph_id_type=}"
    )

    normalized_graph_ids = None
    if graph_ids is not None:
        normalized_graph_ids = [graph_id.strip() for graph_id in graph_ids]
    normalized_graph_id_type = None
    if graph_id_type is not None:
        normalized_graph_id_type = graph_id_type.strip()
    normalized_camera_store_id = None
    if camera_store_id is not None:
        normalized_camera_store_id = camera_store_id.strip()

    if normalized_graph_ids is not None:
        assert len(set(normalized_graph_ids)) == len(normalized_graph_ids), (
            "Expected `graph_ids` to be unique after trimming. "
            f"{normalized_graph_ids=}"
        )

    script_source = _build_plotly_camera_sync_script(
        graph_ids=normalized_graph_ids,
        graph_id_type=normalized_graph_id_type,
        camera_store_id=normalized_camera_store_id,
    )
    script_url_path = _build_plotly_camera_sync_script_url(
        graph_ids=normalized_graph_ids,
        graph_id_type=normalized_graph_id_type,
        camera_store_id=normalized_camera_store_id,
    )
    _register_plotly_camera_sync_route(
        app=app,
        script_url_path=script_url_path,
        script_source=script_source,
    )
    _append_dash_script_url(
        app=app,
        script_url_path=script_url_path,
    )
    return script_url_path


def _build_plotly_camera_sync_script(
    graph_ids: Optional[List[str]],
    graph_id_type: Optional[str],
    camera_store_id: Optional[str],
) -> str:
    """Build one browser-side Plotly camera-sync script.

    Args:
        graph_ids: Optional ordered Dash ids for Plotly 3D graph components to
            synchronize.
        graph_id_type: Optional Dash pattern-id `type` for synchronized Plotly
            graph components.
        camera_store_id: Optional Dash `dcc.Store` id that should receive the
            latest synchronized camera state.

    Returns:
        JavaScript source code string for browser-side camera sync.
    """

    def validate_inputs() -> None:
        assert graph_ids is None or isinstance(graph_ids, list), (
            "Expected `graph_ids` to be `None` or a list. " f"{type(graph_ids)=}"
        )
        if graph_ids is not None:
            assert len(graph_ids) >= 2, (
                "Expected at least two graph ids for synchronization. " f"{graph_ids=}"
            )
            for graph_id in graph_ids:
                assert isinstance(graph_id, str), (
                    "Expected each graph id to be a string. " f"{type(graph_id)=}"
                )
                assert graph_id.strip() != "", (
                    "Expected each graph id to be non-empty after trimming. "
                    f"{graph_ids=}"
                )
        assert graph_id_type is None or isinstance(graph_id_type, str), (
            "Expected `graph_id_type` to be `None` or a string. "
            f"{type(graph_id_type)=}"
        )
        if graph_id_type is not None:
            assert graph_id_type.strip() != "", (
                "Expected `graph_id_type` to be non-empty after trimming. "
                f"{graph_id_type=}"
            )
        assert camera_store_id is None or isinstance(camera_store_id, str), (
            "Expected `camera_store_id` to be `None` or a string. "
            f"{type(camera_store_id)=}"
        )
        if camera_store_id is not None:
            assert camera_store_id.strip() != "", (
                "Expected `camera_store_id` to be non-empty after trimming. "
                f"{camera_store_id=}"
            )

    validate_inputs()

    assert (graph_ids is None) != (graph_id_type is None), (
        "Expected exactly one of `graph_ids` or `graph_id_type`. "
        f"{graph_ids=}, {graph_id_type=}"
    )

    normalized_graph_ids = None
    if graph_ids is not None:
        normalized_graph_ids = [graph_id.strip() for graph_id in graph_ids]
    normalized_graph_id_type = None
    if graph_id_type is not None:
        normalized_graph_id_type = graph_id_type.strip()
    normalized_camera_store_id = None
    if camera_store_id is not None:
        normalized_camera_store_id = camera_store_id.strip()

    if normalized_graph_ids is not None:
        assert len(set(normalized_graph_ids)) == len(normalized_graph_ids), (
            "Expected `graph_ids` to be unique after trimming. "
            f"{normalized_graph_ids=}"
        )

    template_source = _load_plotly_camera_sync_template(
        template_path=PLOTLY_CAMERA_SYNC_TEMPLATE_PATH,
    )
    script_source = template_source.replace(
        "__GRAPH_IDS_JSON__",
        json.dumps(normalized_graph_ids),
    )
    script_source = script_source.replace(
        "__GRAPH_ID_TYPE_JSON__",
        json.dumps(normalized_graph_id_type),
    )
    return script_source.replace(
        "__CAMERA_STORE_ID_JSON__",
        json.dumps(normalized_camera_store_id),
    )


def _build_plotly_camera_sync_script_url(
    graph_ids: Optional[List[str]],
    graph_id_type: Optional[str],
    camera_store_id: Optional[str],
) -> str:
    """Build one stable served URL path for one Plotly sync configuration.

    Args:
        graph_ids: Optional ordered Dash ids for Plotly 3D graph components to
            synchronize.
        graph_id_type: Optional Dash pattern-id `type` for synchronized Plotly
            graph components.
        camera_store_id: Optional Dash `dcc.Store` id that should receive the
            latest synchronized camera state.

    Returns:
        Absolute app-local URL path for the generated script.
    """

    def validate_inputs() -> None:
        assert graph_ids is None or isinstance(graph_ids, list), (
            "Expected `graph_ids` to be `None` or a list. " f"{type(graph_ids)=}"
        )
        if graph_ids is not None:
            assert len(graph_ids) >= 2, (
                "Expected at least two graph ids for synchronization. " f"{graph_ids=}"
            )
        assert graph_id_type is None or isinstance(graph_id_type, str), (
            "Expected `graph_id_type` to be `None` or a string. "
            f"{type(graph_id_type)=}"
        )
        assert camera_store_id is None or isinstance(camera_store_id, str), (
            "Expected `camera_store_id` to be `None` or a string. "
            f"{type(camera_store_id)=}"
        )

    validate_inputs()

    config_signature = hashlib.sha1(
        json.dumps(
            {
                "graph_ids": graph_ids,
                "graph_id_type": graph_id_type,
                "camera_store_id": camera_store_id,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"/__shared_plotly_camera_sync/{config_signature}.js"


def _register_plotly_camera_sync_route(
    app: Dash,
    script_url_path: str,
    script_source: str,
) -> None:
    """Register one Flask route that serves the generated camera-sync script.

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

    def serve_plotly_camera_sync_script() -> Response:
        """Serve the generated Plotly camera-sync script.

        Args:
            None.

        Returns:
            JavaScript response for browser loading.
        """

        return Response(script_source, mimetype="text/javascript")

    app.server.add_url_rule(
        rule=script_url_path,
        endpoint=endpoint_name,
        view_func=serve_plotly_camera_sync_script,
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
def _load_plotly_camera_sync_template(
    template_path: Path,
) -> str:
    """Load the shared Plotly camera-sync template from disk.

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

    return template_path.resolve().read_text(encoding="utf-8")
