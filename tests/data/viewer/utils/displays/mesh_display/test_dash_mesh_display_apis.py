"""Tests for the Dash mesh display artifact-path APIs."""

from pathlib import Path
from typing import Callable

import pytest
import torch
from dash import dcc

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
    with_axis = create_display(mesh_path, lock_roll=LOCK_ROLL_AXIS)

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
