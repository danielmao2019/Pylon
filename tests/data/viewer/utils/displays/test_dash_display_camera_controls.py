"""Tests for the trackball camera controls each Dash Plotly display factory renders under, without and with a lock_roll."""

import base64
import json
import math
from typing import Optional, Tuple

import pytest
import torch
from dash import dcc

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import (
    ROLL_LOCKED_GRAPH_ID_TYPE,
)
from data.viewer.utils.displays.mesh.dash.core_mesh_display import (
    create_dash_mesh_display,
)
from data.viewer.utils.displays.points.dash.core_points_display import (
    create_dash_points_display,
)

# Deliberately non-axis-aligned, so nothing passes by coinciding with a world axis.
NON_AXIS_ALIGNED_LOCK_ROLL = (0.3, 0.9, -0.2)
# The unit-length direction of NON_AXIS_ALIGNED_LOCK_ROLL, the axis a display's controls normalize it to.
NORMALIZED_LOCK_ROLL = tuple(
    component / math.sqrt(sum(value * value for value in NON_AXIS_ALIGNED_LOCK_ROLL))
    for component in NON_AXIS_ALIGNED_LOCK_ROLL
)


@pytest.mark.parametrize("display_kind", ["points", "mesh"])
def test_a_display_built_without_lock_roll_renders_the_free_trackball(
    display_kind: str,
) -> None:
    """A Plotly display built with no lock_roll renders under the free-roll orbit dragmode with no camera.up and no component id, the default trackball control.

    Args:
        display_kind: Which Plotly display factory to build, `"points"` or `"mesh"`.

    Returns:
        None.
    """
    display = _build_display(display_kind=display_kind, lock_roll=None)

    scene = display.figure.layout.scene.to_plotly_json()
    assert scene["dragmode"] == "orbit", (
        "Expected the free-trackball display to run the orbit dragmode. "
        f"{display_kind=} {scene=}"
    )
    assert not ("camera" in scene and "up" in scene["camera"]), (
        "Expected the free-trackball display to pin no camera.up. "
        f"{display_kind=} {scene=}"
    )
    assert "id" not in display.to_plotly_json()["props"], (
        "Expected the free-trackball display to carry no component id, so the "
        f"roll-lock callback matches nothing of it. {display_kind=} {display.id=}"
    )
    return


@pytest.mark.parametrize("display_kind", ["points", "mesh"])
def test_a_display_hands_lock_roll_to_its_controls_unchanged(
    display_kind: str,
) -> None:
    """A Plotly display built with a lock_roll renders under the controls that axis builds: the orbit dragmode, the data aspect, the normalized axis as camera.up, and the roll-locked graph id handing the callback that axis.

    Args:
        display_kind: Which Plotly display factory to build, `"points"` or `"mesh"`.

    Returns:
        None.
    """
    display = _build_display(
        display_kind=display_kind, lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    )

    scene = display.figure.layout.scene.to_plotly_json()
    assert scene["dragmode"] == "orbit", (
        "Expected the roll-locked display to run the orbit dragmode. "
        f"{display_kind=} {scene=}"
    )
    assert scene["aspectmode"] == "data", (
        "Expected the roll-locked display to draw the scene at its data's own "
        f"proportions. {display_kind=} {scene=}"
    )
    up = scene["camera"]["up"]
    assert up == pytest.approx(
        {
            "x": NORMALIZED_LOCK_ROLL[0],
            "y": NORMALIZED_LOCK_ROLL[1],
            "z": NORMALIZED_LOCK_ROLL[2],
        },
        abs=1e-12,
    ), (
        "Expected the roll-locked display's camera.up to be the normalized "
        f"lock_roll. {display_kind=} {up=} {NORMALIZED_LOCK_ROLL=}"
    )
    assert display.id["type"] == ROLL_LOCKED_GRAPH_ID_TYPE, (
        "Expected the roll-locked display to carry the graph id type the roll-lock "
        f"callback matches. {display_kind=} {display.id=}"
    )
    graph_axis = json.loads(base64.b64decode(display.id["lock_roll"]))
    assert graph_axis == pytest.approx(NORMALIZED_LOCK_ROLL, abs=1e-12), (
        "Expected the roll-locked display's graph id to hand the callback the "
        f"normalized lock_roll. {display_kind=} {graph_axis=} {NORMALIZED_LOCK_ROLL=}"
    )
    return


def _build_display(
    display_kind: str,
    lock_roll: Optional[Tuple[float, float, float]],
) -> dcc.Graph:
    """Build one Plotly display of the given kind over hand-built synthetic geometry, handing lock_roll to its factory.

    Args:
        display_kind: Which Plotly display factory to build, `"points"` or `"mesh"`.
        lock_roll: Axis handed to the factory unchanged, as a non-zero `(x, y, z)` world-space direction, or None for the free trackball.

    Returns:
        The Dash `dcc.Graph` the factory built.
    """

    def _validate_inputs() -> None:
        assert display_kind in ("points", "mesh"), (
            "Expected `display_kind` to name a Plotly display factory. "
            f"{display_kind=}"
        )

    _validate_inputs()

    if display_kind == "points":
        point_cloud = PointCloud(
            xyz=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
        )
        display = create_dash_points_display(
            point_cloud=point_cloud, lock_roll=lock_roll
        )
    else:
        mesh = Mesh(
            verts=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            faces=torch.tensor(
                [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int64
            ),
            texture=MeshTextureVertexColor(
                vertex_color=torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
            ),
        )
        display = create_dash_mesh_display(mesh=mesh, lock_roll=lock_roll)
    return display
