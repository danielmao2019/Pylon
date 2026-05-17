"""Shared Flask routes and iframe URL builders."""

from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from dash import Dash
from flask import abort, request, send_file
from data.viewer.utils.reusable_viewers.common.files import resolve_allowed_file
from data.viewer.utils.reusable_viewers.paths import STATIC_ROOT, get_shared_asset_path


def build_pointcloud_url(file_path: Optional[Path], label: str) -> str:
    """Build a point-cloud iframe URL.

    Args:
        file_path: Optional PLY artifact path to display.
        label: Human-readable panel label.

    Returns:
        Iframe URL for the shared point-cloud page.
    """
    assert file_path is None or isinstance(
        file_path, Path
    ), "File path must be None or a Path. file_path=%r" % (file_path,)
    assert isinstance(label, str), "Label must be a string. label=%r" % label
    assert label, "Label must be non-empty. label=%r" % label
    query = {"label": label}
    if file_path is None:
        query["missing"] = "1"
    else:
        query["file"] = str(file_path)
    return "/shared-static/pointcloud_viewer.html?%s" % urlencode(query)


def register_common_routes(app: Dash) -> None:
    """Register shared static and artifact-serving routes.

    Args:
        app: Dash app whose Flask server receives the routes.

    Returns:
        None.
    """
    assert isinstance(app, Dash), "App must be a Dash instance. app=%r" % (app,)
    if getattr(app.server, "_shared_viewer_common_routes_registered", False):
        return None
    assert STATIC_ROOT.exists(), "Shared static root is missing. root=%s" % STATIC_ROOT

    @app.server.route("/shared-static/<path:filename>")
    def _serve_shared_static(filename: str):
        """Serve shared viewer static assets.

        Args:
            filename: Static filename under the shared viewer static root.

        Returns:
            Flask file response for the static asset.
        """
        assert isinstance(filename, str), "Filename must be a string. filename=%r" % (
            filename,
        )
        static_path = (STATIC_ROOT / filename).resolve()
        if not static_path.is_file():
            abort(404)
        return send_file(static_path)

    @app.server.route("/shared-assets/<path:filename>")
    def _serve_shared_asset(filename: str):
        """Serve shared viewer support assets.

        Args:
            filename: Asset filename under the shared viewer assets root.

        Returns:
            Flask file response for the static asset.
        """
        assert isinstance(filename, str), "Filename must be a string. filename=%r" % (
            filename,
        )
        asset_path = get_shared_asset_path(filename=filename)
        if not asset_path.is_file():
            abort(404)
        return send_file(asset_path)

    @app.server.route("/shared-files")
    def _serve_shared_file():
        """Serve an allowed benchmark artifact.

        Args:
            None.

        Returns:
            Flask file response for the requested artifact.
        """
        path_value = request.args.get("path", "")
        try:
            resolved_path = resolve_allowed_file(path_value)
        except AssertionError:
            abort(404)
        return send_file(resolved_path)

    app.server._shared_viewer_common_routes_registered = True
    return None
