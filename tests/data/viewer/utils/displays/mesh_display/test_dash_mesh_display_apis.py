"""Tests for the Dash mesh display artifact-path APIs."""

import re
from pathlib import Path
from typing import Callable

import pytest
import torch
from dash import Dash, dcc

from data.structures.three_d.mesh import Mesh, MeshTextureVertexColor
from data.viewer.utils.displays.mesh.dash.apis import (
    create_color_mesh_display,
    create_heatmap_mesh_display,
    create_segmentation_mesh_display,
)

MESH_DISPLAY_APIS = [
    create_color_mesh_display,
    create_segmentation_mesh_display,
    create_heatmap_mesh_display,
]
MESH_DISPLAY_API_IDS = [
    "create_color_mesh_display",
    "create_segmentation_mesh_display",
    "create_heatmap_mesh_display",
]
LOCK_ROLL_AXIS = (0.3, 0.0, 0.95)
LOCK_ROLL_AXIS_NORMALIZED = {
    "x": 0.3011313679370974,
    "y": 0.0,
    "z": 0.9535826651341417,
}
# Component id the roll-locked displays below are built under, and therefore the id
# the registration each of them performs must address.
LOCKED_GRAPH_ID = "locked-mesh-graph"
# The rejection each missing half of the lock target must be named by. Matching the
# display factory's own wording rather than the word `app` or `graph_id` alone is what
# keeps these clauses from passing on the registration helper's type assertions, which
# report a wrong type instead of naming what the lock is missing.
MISSING_APP_MESSAGE = re.escape("Expected an `app` alongside `lock_roll`")
MISSING_GRAPH_ID_MESSAGE = re.escape("Expected a `graph_id` alongside `lock_roll`")
# The rejection an `app` handed in with no axis must be named by. Matching this
# contract's own wording rather than the word `app` or `lock_roll` alone is what keeps
# the clause from passing on some unrelated later failure that happens to mention one
# of those words instead of naming what the lock is missing.
MISSING_LOCK_ROLL_MESSAGE = re.escape("Expected a `lock_roll` alongside `app`")


def _write_mesh_artifact(tmp_path: Path) -> str:
    """Write one small vertex-colored mesh artifact readable by all three display APIs.

    Args:
        tmp_path: Pytest-provided temporary directory the artifact is written into.

    Returns:
        On-disk path of the written OBJ artifact, whose channel-0 vertex-color values double as segmentation class ids and as non-negative heatmap scalars.
    """

    mesh = Mesh(
        verts=torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        texture=MeshTextureVertexColor(
            vertex_color=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        ),
    )
    mesh_path = tmp_path / "mesh.obj"
    mesh.save(path=mesh_path)
    return str(mesh_path)


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_forwards_lock_roll_to_the_scene(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Seed `layout.scene.camera.up` from a supplied `lock_roll` axis.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    display = create_display(
        _write_mesh_artifact(tmp_path=tmp_path),
        lock_roll=LOCK_ROLL_AXIS,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    )

    scene = display.figure.layout.scene
    assert scene.dragmode == "orbit", (
        "Expected a roll-locked mesh display to render gl3d `dragmode` "
        f'"orbit". {scene.dragmode=}'
    )
    assert scene.camera.up.x == pytest.approx(LOCK_ROLL_AXIS_NORMALIZED["x"]), (
        "Expected the supplied `lock_roll` axis to reach the rendered "
        f"`camera.up`. {scene.camera.up=} {LOCK_ROLL_AXIS=}"
    )
    assert scene.camera.up.y == pytest.approx(LOCK_ROLL_AXIS_NORMALIZED["y"]), (
        "Expected the supplied `lock_roll` axis to reach the rendered "
        f"`camera.up`. {scene.camera.up=} {LOCK_ROLL_AXIS=}"
    )
    assert scene.camera.up.z == pytest.approx(LOCK_ROLL_AXIS_NORMALIZED["z"]), (
        "Expected the supplied `lock_roll` axis to reach the rendered "
        f"`camera.up`. {scene.camera.up=} {LOCK_ROLL_AXIS=}"
    )


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_without_lock_roll_pins_no_axis(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Render the pre-`lock_roll` scene, an unpinned orbit camera, when the axis is omitted.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    display = create_display(_write_mesh_artifact(tmp_path=tmp_path))

    assert display.figure.layout.scene.to_plotly_json() == {"dragmode": "orbit"}, (
        "Expected an omitted `lock_roll` to render the unpinned orbit scene "
        "the display rendered before the parameter existed. "
        f"{display.figure.layout.scene.to_plotly_json()=}"
    )


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_lock_roll_changes_only_the_camera(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Confine the effect of `lock_roll` to `layout.scene.camera`, leaving the traces and the rest of the layout untouched.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    mesh_path = _write_mesh_artifact(tmp_path=tmp_path)
    without_axis = create_display(mesh_path)
    with_axis = create_display(
        mesh_path,
        lock_roll=LOCK_ROLL_AXIS,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    )

    without_axis_figure = without_axis.figure.to_plotly_json()
    with_axis_figure = with_axis.figure.to_plotly_json()
    assert without_axis_figure["data"] == with_axis_figure["data"], (
        "Expected `lock_roll` to leave the rendered traces untouched. "
        f"{without_axis_figure['data']=} {with_axis_figure['data']=}"
    )
    with_axis_figure["layout"]["scene"].pop("camera")
    assert without_axis_figure["layout"] == with_axis_figure["layout"], (
        "Expected `lock_roll` to touch nothing in the layout but "
        "`scene.camera`. "
        f"{without_axis_figure['layout']=} {with_axis_figure['layout']=}"
    )


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_forwards_the_lock_to_the_app(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Register the roll lock on the supplied app against the supplied graph id, so an artifact-path caller needs no second call of its own.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    app = Dash(__name__)

    display = create_display(
        _write_mesh_artifact(tmp_path=tmp_path),
        lock_roll=LOCK_ROLL_AXIS,
        app=app,
        graph_id=LOCKED_GRAPH_ID,
    )

    registrations = [
        registration
        for registration in app._callback_list
        if registration["clientside_function"] is not None
    ]
    assert len(registrations) == 1, (
        "An artifact-path API given a roll-lock axis must forward the app it was "
        "handed, so the lock ends up registered rather than dropped on the way. "
        f"{registrations=}"
    )
    assert registrations[0]["inputs"] == [
        {"id": LOCKED_GRAPH_ID, "property": "relayoutData"}
    ], (
        "An artifact-path API must forward the graph id it was handed, so the lock "
        f"addresses the graph it returned. {registrations[0]=} {LOCKED_GRAPH_ID=}"
    )
    assert display.id == LOCKED_GRAPH_ID, (
        "An artifact-path API must forward the graph id onto the graph it returns, "
        f"which is what the lock addresses. {display.to_plotly_json()['props'].keys()=} "
        f"{LOCKED_GRAPH_ID=}"
    )


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_rejects_an_axis_without_the_lock_target(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Reject a supplied axis that names neither the app nor the graph the lock would be registered on, rather than returning a display that only looks locked.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    mesh_path = _write_mesh_artifact(tmp_path=tmp_path)

    with pytest.raises(AssertionError, match=MISSING_APP_MESSAGE):
        create_display(mesh_path, lock_roll=LOCK_ROLL_AXIS, graph_id=LOCKED_GRAPH_ID)
    with pytest.raises(AssertionError, match=MISSING_GRAPH_ID_MESSAGE):
        create_display(mesh_path, lock_roll=LOCK_ROLL_AXIS, app=Dash(__name__))


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_rejects_an_app_without_the_axis(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Reject an app handed in with no axis for it to lock about, rather than returning the unlocked display a caller who believed they were locking would otherwise get.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    mesh_path = _write_mesh_artifact(tmp_path=tmp_path)

    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_MESSAGE):
        create_display(mesh_path, app=Dash(__name__))
    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_MESSAGE):
        create_display(mesh_path, app=Dash(__name__), graph_id=LOCKED_GRAPH_ID)


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_graph_id_without_the_axis_names_the_unlocked_graph(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Allow a graph id named with no axis and carry it onto the returned graph, because naming a graph is useful on its own and only the app is meaningless without an axis.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    display = create_display(
        _write_mesh_artifact(tmp_path=tmp_path),
        graph_id=LOCKED_GRAPH_ID,
    )

    assert display.id == LOCKED_GRAPH_ID, (
        "Expected a graph id named without an axis to reach the graph the API "
        f"returns. {display.to_plotly_json()['props'].keys()=} {LOCKED_GRAPH_ID=}"
    )
    assert display.figure.layout.scene.to_plotly_json() == {"dragmode": "orbit"}, (
        "Expected a graph id named without an axis to leave the display unlocked, "
        "since an id names the graph and locks nothing. "
        f"{display.figure.layout.scene.to_plotly_json()=}"
    )


@pytest.mark.parametrize(
    "create_display",
    MESH_DISPLAY_APIS,
    ids=MESH_DISPLAY_API_IDS,
)
def test_mesh_display_api_naming_neither_returns_the_unlocked_graph_unchanged(
    create_display: Callable[..., dcc.Graph],
    tmp_path: Path,
) -> None:
    """Return byte-for-byte the unlocked graph the API returns when the three roll-lock parameters are omitted entirely.

    Args:
        create_display: Mesh display API under test, called with an artifact path.
        tmp_path: Pytest-provided temporary directory the mesh artifact is written into.

    Returns:
        None.
    """

    mesh_path = _write_mesh_artifact(tmp_path=tmp_path)
    omitted = create_display(mesh_path)
    explicit_none = create_display(mesh_path, lock_roll=None, app=None, graph_id=None)

    assert omitted.to_plotly_json() == explicit_none.to_plotly_json(), (
        "Expected passing the roll-lock parameters as None to render exactly what "
        f"omitting them renders. {omitted.to_plotly_json()=} "
        f"{explicit_none.to_plotly_json()=}"
    )
    assert set(omitted.to_plotly_json()["props"]) == {"figure"}, (
        "Expected a display named neither an axis nor an app to return a graph "
        "carrying nothing but its figure, which is what every existing call site "
        f"already gets. {omitted.to_plotly_json()['props'].keys()=}"
    )
