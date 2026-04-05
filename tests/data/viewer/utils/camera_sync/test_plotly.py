"""Tests for shared Plotly camera-sync utilities."""

import pytest
from dash import Dash, html

from data.viewer.utils.camera_sync import register_plotly_camera_sync
from data.viewer.utils.camera_sync.plotly import _build_plotly_camera_sync_script


def test_register_plotly_camera_sync_serves_script_and_appends_resource() -> None:
    """Register one served script resource for Plotly camera sync.

    Args:
        None.

    Returns:
        None.
    """

    app = Dash(__name__)
    script_url_path = register_plotly_camera_sync(
        app=app,
        graph_ids=[" graph-a ", "graph-b", "graph-c "],
    )

    assert isinstance(script_url_path, str), f"{type(script_url_path)=}"
    assert script_url_path.startswith(
        "/__shared_plotly_camera_sync/"
    ), f"{script_url_path=}"
    app.layout = html.Div()
    response = app.server.test_client().get(script_url_path)
    assert response.status_code == 200, f"{response.status_code=}"
    response_text = response.get_data(as_text=True)
    assert 'const GRAPH_IDS = ["graph-a", "graph-b", "graph-c"];' in response_text
    assert (
        script_url_path in app.config.external_scripts
    ), f"{app.config.external_scripts=}"
    assert script_url_path in app.index(), f"{app.index()=}"


def test_build_plotly_camera_sync_script_embeds_graph_ids() -> None:
    """Embed the requested graph ids in the generated script.

    Args:
        None.

    Returns:
        None.
    """

    script_source = _build_plotly_camera_sync_script(
        graph_ids=["graph-a", "graph-b"],
        graph_id_type=None,
        camera_store_id=None,
    )

    assert isinstance(script_source, str), f"{type(script_source)=}"
    assert 'const GRAPH_IDS = ["graph-a", "graph-b"];' in script_source
    assert "const GRAPH_ID_TYPE = null;" in script_source
    assert "const CAMERA_STORE_ID = null;" in script_source
    assert "window.setInterval(attachSync, 500);" in script_source
    assert "__GRAPH_IDS_JSON__" not in script_source, f"{script_source[:240]=}"


def test_build_plotly_camera_sync_script_embeds_pattern_graph_type() -> None:
    """Embed the requested Dash pattern-id type in the generated script.

    Args:
        None.

    Returns:
        None.
    """

    script_source = _build_plotly_camera_sync_script(
        graph_ids=None,
        graph_id_type=" point-cloud-graph ",
        camera_store_id="camera-state",
    )

    assert "const GRAPH_IDS = null;" in script_source
    assert 'const GRAPH_ID_TYPE = "point-cloud-graph";' in script_source
    assert 'const CAMERA_STORE_ID = "camera-state";' in script_source
    assert "querySelectorAll(\"[id]\")" in script_source


def test_build_plotly_camera_sync_script_embeds_camera_store_id() -> None:
    """Embed the optional camera-store id in the generated script.

    Args:
        None.

    Returns:
        None.
    """

    script_source = _build_plotly_camera_sync_script(
        graph_ids=["graph-a", "graph-b"],
        graph_id_type=None,
        camera_store_id=" shared-camera-store ",
    )

    assert 'const CAMERA_STORE_ID = "shared-camera-store";' in script_source
    assert "window.dash_clientside.set_props(CAMERA_STORE_ID" in script_source


def test_register_plotly_camera_sync_rejects_duplicate_graph_ids() -> None:
    """Reject duplicate graph ids after normalization.

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError):
        register_plotly_camera_sync(
            app=Dash(__name__),
            graph_ids=["graph-a", " graph-a "],
        )


def test_register_plotly_camera_sync_rejects_missing_target_mode() -> None:
    """Reject registration when neither graph-id mode is specified.

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError):
        register_plotly_camera_sync(app=Dash(__name__))


def test_register_plotly_camera_sync_rejects_multiple_target_modes() -> None:
    """Reject registration when both graph-id modes are specified.

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError):
        register_plotly_camera_sync(
            app=Dash(__name__),
            graph_ids=["graph-a", "graph-b"],
            graph_id_type="point-cloud-graph",
        )
