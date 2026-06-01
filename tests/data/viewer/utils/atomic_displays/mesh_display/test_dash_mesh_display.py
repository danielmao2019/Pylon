"""Tests for the Dash mesh display style-arg functions."""

import plotly.graph_objects as go
import pytest
import torch
from dash import dcc

from data.structures.three_d.mesh import Mesh, MeshTextureVertexColor
from data.viewer.utils.atomic_displays.mesh.dash.core_mesh_display import (
    DEFAULT_MESH_COLOR,
    DEFAULT_MESH_OPACITY,
    DEFAULT_MESH_SIDE,
    create_dash_mesh_display,
    create_dash_mesh_scene,
)


def _build_vertex_color_mesh() -> Mesh:
    """Build one small vertex-colored mesh for Dash display tests.

    Args:
        None.

    Returns:
        Vertex-colored mesh with three RGB-tagged vertices.
    """

    return Mesh(
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


def test_create_dash_mesh_scene_returns_mesh3d() -> None:
    """Build one Plotly `go.Mesh3d` trace from a vertex-color mesh.

    Args:
        None.

    Returns:
        None.
    """

    scene = create_dash_mesh_scene(mesh=_build_vertex_color_mesh())

    assert isinstance(scene, go.Mesh3d), f"{type(scene)=}"
    assert scene.type == "mesh3d", f"{scene.type=}"
    assert tuple(float(value) for value in scene.x) == (
        1.0,
        3.0,
        1.0,
    ), f"{scene.x=}"
    assert tuple(int(value) for value in scene.i) == (0,), f"{scene.i=}"
    assert tuple(int(value) for value in scene.j) == (1,), f"{scene.j=}"
    assert tuple(int(value) for value in scene.k) == (2,), f"{scene.k=}"


def test_create_dash_mesh_scene_default_opacity() -> None:
    """Default the trace opacity to `DEFAULT_MESH_OPACITY` when None.

    Args:
        None.

    Returns:
        None.
    """

    scene = create_dash_mesh_scene(mesh=_build_vertex_color_mesh())

    assert scene.opacity == pytest.approx(DEFAULT_MESH_OPACITY), (
        "Expected the Dash mesh scene to default opacity to "
        "`DEFAULT_MESH_OPACITY` when no override is supplied. "
        f"{scene.opacity=} {DEFAULT_MESH_OPACITY=}"
    )


def test_create_dash_mesh_scene_explicit_opacity() -> None:
    """Set the trace opacity from an explicit `mesh_opacity`.

    Args:
        None.

    Returns:
        None.
    """

    scene = create_dash_mesh_scene(
        mesh=_build_vertex_color_mesh(),
        mesh_opacity=0.25,
    )

    assert scene.opacity == pytest.approx(0.25), f"{scene.opacity=}"


def test_create_dash_mesh_scene_explicit_color_is_uniform() -> None:
    """Set a uniform color and drop per-vertex colors when `mesh_color` is given.

    Args:
        None.

    Returns:
        None.
    """

    scene = create_dash_mesh_scene(
        mesh=_build_vertex_color_mesh(),
        mesh_color="#abcdef",
    )

    assert scene.color == "#abcdef", f"{scene.color=}"
    assert scene.vertexcolor is None, (
        "Expected an explicit `mesh_color` to suppress per-vertex coloring. "
        f"{scene.vertexcolor=}"
    )


def test_create_dash_mesh_scene_default_uses_per_vertex_rgb() -> None:
    """Use per-vertex RGB when `mesh_color` is None and colors are present.

    Args:
        None.

    Returns:
        None.
    """

    scene = create_dash_mesh_scene(mesh=_build_vertex_color_mesh())

    assert scene.color is None, (
        "Expected no uniform color when `mesh_color` is None and the mesh "
        f"carries per-vertex colors. {scene.color=}"
    )
    assert scene.vertexcolor == (
        "rgb(255,0,0)",
        "rgb(0,255,0)",
        "rgb(0,0,255)",
    ), f"{scene.vertexcolor=}"


def test_create_dash_mesh_display_returns_graph() -> None:
    """Wrap the mesh scene in a Dash `dcc.Graph`.

    Args:
        None.

    Returns:
        None.
    """

    display = create_dash_mesh_display(mesh=_build_vertex_color_mesh())

    assert isinstance(display, dcc.Graph), f"{type(display)=}"
    assert isinstance(display.figure, go.Figure), f"{type(display.figure)=}"
    assert len(display.figure.data) == 1, f"{len(display.figure.data)=}"
    assert display.figure.data[0].type == "mesh3d", f"{display.figure.data[0].type=}"


def test_default_mesh_style_constants() -> None:
    """Pin the lib-owned mesh style defaults.

    Args:
        None.

    Returns:
        None.
    """

    assert DEFAULT_MESH_COLOR == "#cccccc", f"{DEFAULT_MESH_COLOR=}"
    assert DEFAULT_MESH_OPACITY == pytest.approx(1.0), f"{DEFAULT_MESH_OPACITY=}"
    assert DEFAULT_MESH_SIDE == "double", f"{DEFAULT_MESH_SIDE=}"
