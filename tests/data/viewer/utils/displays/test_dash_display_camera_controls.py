"""Tests that every Dash 3D display renders the camera controls its caller asked for.

The camera controls a caller names through `lock_roll` are inert unless they reach
the layout of the figure the display factory actually returns, so every assertion
here reads the rendered `figure.layout` rather than the factory's inputs.
"""

import math
import re
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
from dash import Dash, dcc

from data.structures.three_d.mesh import Mesh, MeshTextureVertexColor
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.displays.mesh.dash.core_mesh_display import (
    create_dash_mesh_display,
)
from data.viewer.utils.displays.points.dash.core_points_display import (
    create_dash_points_display,
)

# Deliberately non-axis-aligned, so nothing can pass by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# Component id the roll-locked display below is built under, and therefore the id the
# registration it performs must address.
LOCKED_GRAPH_ID = "locked-display-graph"
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


def build_dash_mesh_display(**kwargs: Any) -> dcc.Graph:
    """Build the Dash mesh display over one small vertex-colored mesh.

    Args:
        **kwargs: Display arguments forwarded to `create_dash_mesh_display`.

    Returns:
        Dash `dcc.Graph` wrapping the mesh scene.
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
    return create_dash_mesh_display(mesh=mesh, **kwargs)


def build_dash_points_display(**kwargs: Any) -> dcc.Graph:
    """Build the Dash point-cloud display over one small point cloud.

    Args:
        **kwargs: Display arguments forwarded to `create_dash_points_display`.

    Returns:
        Dash `dcc.Graph` wrapping the point-cloud scene.
    """
    point_cloud = PointCloud(
        xyz=torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    return create_dash_points_display(point_cloud=point_cloud, **kwargs)


DASH_3D_DISPLAY_FACTORIES: Tuple[Tuple[str, Callable[..., dcc.Graph]], ...] = (
    ("mesh", build_dash_mesh_display),
    ("points", build_dash_points_display),
)


def clientside_registrations(app: Dash) -> List[Dict[str, Any]]:
    """Collect the clientside callbacks registered on a Dash app.

    Args:
        app: Dash app whose callback registrations are read.

    Returns:
        List of the app's callback registration dicts whose
        `clientside_function` is set, each carrying an `inputs` list of
        `{"id", "property"}` dicts.
    """
    return [
        registration
        for registration in app._callback_list
        if registration["clientside_function"] is not None
    ]


def expected_camera_up(lock_roll: Tuple[float, float, float]) -> Dict[str, float]:
    """Compute the unit-length Plotly camera up vector a lock axis must produce.

    Args:
        lock_roll: Axis to lock camera roll about, as a non-zero `(x, y, z)`
            world-space direction of any length.

    Returns:
        Dict with `x`, `y`, and `z` unit-length components.
    """
    length = math.sqrt(sum(component * component for component in lock_roll))
    return {
        "x": lock_roll[0] / length,
        "y": lock_roll[1] / length,
        "z": lock_roll[2] / length,
    }


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_no_axis_renders_the_free_roll_dragmode(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """A display given no lock_roll renders the dragmode under which roll is reachable, never the pose-clamping one Plotly runs when the layout names no dragmode."""
    display = build_display(lock_roll=None)

    scene = display.figure.layout.scene
    assert scene.dragmode != "turntable", (
        "plotly.js pins the camera up vector to (0, 0, 1) under the turntable "
        "dragmode, so a display that renders turntable is roll-locked about world "
        f"+Z, an axis its caller never named. {display_kind=} {scene.dragmode=}"
    )
    assert scene.dragmode == "orbit", (
        "A display given no roll-lock axis must render the Plotly dragmode that "
        "leaves the camera up vector free to tilt, which is what an unlocked "
        f"display means. {display_kind=} {scene.dragmode=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_a_supplied_axis_renders_a_roll_locked_camera(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """A display given a lock_roll renders the roll-locking dragmode with the caller's axis as the camera up vector."""
    display = build_display(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    )

    scene = display.figure.layout.scene
    assert scene.dragmode != "turntable", (
        "plotly.js discards any camera up vector whose normalized z falls below 0.999 "
        "under the turntable dragmode and substitutes (0, 0, 1), so a display that "
        "renders turntable shows the unlocked camera for every axis more than ~2.5 "
        f"degrees off world +Z. {display_kind=} {scene.dragmode=}"
    )
    assert scene.dragmode == "orbit", (
        "A display given a roll-lock axis must render the Plotly dragmode that "
        f"carries that axis through re-render. {display_kind=} {scene.dragmode=}"
    )
    up = expected_camera_up(NON_AXIS_ALIGNED_LOCK_ROLL)
    assert (scene.camera.up.x, scene.camera.up.y, scene.camera.up.z) == (
        up["x"],
        up["y"],
        up["z"],
    ), (
        "A display given a roll-lock axis must carry that axis into the rendered "
        f"camera up vector. {display_kind=} {scene.camera.up=} {up=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_the_two_roll_settings_render_different_cameras(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """The rendered camera differs between the locked and unlocked settings, so the axis cannot be silently discarded."""
    free_scene = build_display(lock_roll=None).figure.layout.scene
    locked_scene = build_display(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    ).figure.layout.scene

    assert free_scene != locked_scene, (
        "Naming a roll-lock axis must change the rendered camera. "
        f"{display_kind=} {free_scene=} {locked_scene=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_an_axis_alone_locks_the_display(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """A display handed an axis registers the roll lock itself, so a caller that names the axis is not left with an unlocked panel for want of a second call it had to know about."""
    app = Dash(__name__)

    build_display(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=app,
        graph_id=LOCKED_GRAPH_ID,
    )

    registrations = clientside_registrations(app=app)
    assert len(registrations) == 1, (
        "A display given a roll-lock axis must put the lock on the app itself, "
        "since seeding the rendered camera up vector holds roll through re-render "
        "but never through a drag. "
        f"{display_kind=} {registrations=}"
    )
    assert registrations[0]["inputs"] == [
        {"id": LOCKED_GRAPH_ID, "property": "relayoutData"}
    ], (
        "The lock a display registers must be driven by that display's own graph, "
        "which is how every camera change on it reaches the lock. "
        f"{display_kind=} {registrations[0]=} {LOCKED_GRAPH_ID=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_the_locked_display_graph_carries_the_registered_id(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """The graph the factory returns carries the supplied id, without which the lock it registered addresses nothing."""
    display = build_display(
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
        app=Dash(__name__),
        graph_id=LOCKED_GRAPH_ID,
    )

    assert display.id == LOCKED_GRAPH_ID, (
        "The lock is registered against the supplied graph id, so the graph the "
        "factory returns must be the one carrying it. "
        f"{display_kind=} {display.to_plotly_json()['props'].keys()=} "
        f"{LOCKED_GRAPH_ID=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_an_axis_without_the_app_is_rejected(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """Naming an axis with no app to register the lock on is rejected, rather than silently rendering a panel that only looks locked."""
    with pytest.raises(AssertionError, match=MISSING_APP_MESSAGE):
        build_display(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            graph_id=LOCKED_GRAPH_ID,
        )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_an_axis_without_the_graph_id_is_rejected(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """Naming an axis with no graph id for the lock to address is rejected, rather than silently rendering a panel that only looks locked."""
    with pytest.raises(AssertionError, match=MISSING_GRAPH_ID_MESSAGE):
        build_display(
            lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL,
            app=Dash(__name__),
        )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_no_axis_leaves_the_returned_graph_where_it_was(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """A display named no axis, no app, and no graph id returns the graph it returned before those parameters existed, so no existing call site moves."""
    display = build_display(lock_roll=None)

    assert set(display.to_plotly_json()["props"]) == {"figure"}, (
        "An unlocked display must return a graph carrying nothing but its figure, "
        "which is what every existing call site already gets. "
        f"{display_kind=} {display.to_plotly_json()['props'].keys()=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_an_app_without_the_axis_is_rejected(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """Handing in an app with no axis for it to lock about is rejected, rather than silently returning the unlocked display a caller who believed they were locking would otherwise get."""
    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_MESSAGE):
        build_display(app=Dash(__name__))
    with pytest.raises(AssertionError, match=MISSING_LOCK_ROLL_MESSAGE):
        build_display(app=Dash(__name__), graph_id=LOCKED_GRAPH_ID)


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_a_graph_id_without_the_axis_names_the_unlocked_graph(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """A graph id named with no axis is allowed and names the returned graph, because naming a graph is useful on its own and only the app is meaningless without an axis."""
    display = build_display(graph_id=LOCKED_GRAPH_ID)

    assert display.id == LOCKED_GRAPH_ID, (
        "A graph id named without an axis must reach the graph the factory "
        f"returns. {display_kind=} {display.to_plotly_json()['props'].keys()=} "
        f"{LOCKED_GRAPH_ID=}"
    )
    scene = display.figure.layout.scene
    assert (scene.camera.up.x, scene.camera.up.y, scene.camera.up.z) == (
        None,
        None,
        None,
    ), (
        "A graph id named without an axis must leave the display unlocked, since "
        f"an id names the graph and locks nothing. {display_kind=} {scene.camera.up=}"
    )


@pytest.mark.parametrize(
    "display_kind, build_display",
    DASH_3D_DISPLAY_FACTORIES,
    ids=[display_kind for display_kind, _ in DASH_3D_DISPLAY_FACTORIES],
)
def test_naming_neither_returns_the_unlocked_component_unchanged(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """Naming neither an axis nor an app returns byte-for-byte the unlocked component the factory returns when the three parameters are omitted entirely."""
    omitted = build_display()
    explicit_none = build_display(lock_roll=None, app=None, graph_id=None)

    assert omitted.to_plotly_json() == explicit_none.to_plotly_json(), (
        "Passing the roll-lock parameters as None must render exactly what "
        f"omitting them renders. {display_kind=} {omitted.to_plotly_json()=} "
        f"{explicit_none.to_plotly_json()=}"
    )
    assert set(omitted.to_plotly_json()["props"]) == {"figure"}, (
        "A display named neither an axis nor an app must return a graph carrying "
        "nothing but its figure, which is what every existing call site already "
        f"gets. {display_kind=} {omitted.to_plotly_json()['props'].keys()=}"
    )
