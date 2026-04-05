"""Application factory for the dataset viewer Dash app."""

from typing import Any

from dash import Dash

from data.viewer.dataset.callbacks import register_viewer_callbacks
from data.viewer.dataset.context import DatasetViewerContext, set_viewer_context
from data.viewer.dataset.layout import build_layout
from data.viewer.utils.camera_sync import register_plotly_camera_sync


def create_app(viewer: Any) -> Dash:
    # Input validations
    assert hasattr(viewer, "backend"), f"viewer must expose backend, got {type(viewer)}"
    assert hasattr(
        viewer, "available_datasets"
    ), f"viewer must expose available_datasets, got {type(viewer)}"

    context = DatasetViewerContext(
        backend=viewer.backend,
        available_datasets=viewer.available_datasets,
    )
    set_viewer_context(context)

    app = Dash(
        __name__,
        title="Dataset Viewer",
        suppress_callback_exceptions=True,
        prevent_initial_callbacks="initial_duplicate",
    )
    register_plotly_camera_sync(
        app=app,
        graph_id_type="point-cloud-graph",
        camera_store_id="camera-state",
    )
    build_layout(app=app)
    register_viewer_callbacks(app=app, viewer=viewer)
    return app
