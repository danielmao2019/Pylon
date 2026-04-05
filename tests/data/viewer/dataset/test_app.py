"""Tests for the dataset viewer Dash app factory."""

from types import SimpleNamespace

from dash import html


class _DummyViewer:
    """Minimal dataset-viewer stub for app-factory tests.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self) -> None:
        """Initialize one minimal viewer stub.

        Args:
            None.

        Returns:
            None.
        """

        self.backend = object()
        self.available_datasets = []


def test_create_app_registers_shared_plotly_camera_sync(monkeypatch) -> None:
    """Register the shared Plotly camera-sync resource on the dataset app.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """

    import data.viewer.dataset.app as dataset_app_module

    monkeypatch.setattr(
        dataset_app_module,
        "DatasetViewerContext",
        lambda backend, available_datasets: SimpleNamespace(
            backend=backend,
            available_datasets=available_datasets,
        ),
    )
    monkeypatch.setattr(dataset_app_module, "set_viewer_context", lambda context: None)
    monkeypatch.setattr(
        dataset_app_module,
        "build_layout",
        lambda app: setattr(app, "layout", html.Div()),
    )
    monkeypatch.setattr(
        dataset_app_module,
        "register_viewer_callbacks",
        lambda app, viewer: None,
    )
    app = dataset_app_module.create_app(viewer=_DummyViewer())

    assert isinstance(app.layout, html.Div), f"{type(app.layout)=}"
    assert any(
        isinstance(script_url, str)
        and script_url.startswith("/__shared_plotly_camera_sync/")
        for script_url in app.config.external_scripts
    ), f"{app.config.external_scripts=}"

    script_url_path = next(
        script_url
        for script_url in app.config.external_scripts
        if isinstance(script_url, str)
        and script_url.startswith("/__shared_plotly_camera_sync/")
    )
    response = app.server.test_client().get(script_url_path)
    response_text = response.get_data(as_text=True)

    assert response.status_code == 200, f"{response.status_code=}"
    assert 'const GRAPH_ID_TYPE = "point-cloud-graph";' in response_text
    assert 'const CAMERA_STORE_ID = "camera-state";' in response_text
    assert script_url_path in app.index(), f"{app.index()=}"
