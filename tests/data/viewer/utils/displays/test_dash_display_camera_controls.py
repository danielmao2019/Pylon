"""Tests that every Dash 3D display renders the camera controls its caller asked for.

The camera controls a caller names through `lock_roll` are inert unless they reach
the layout of the figure the display factory actually returns, so every assertion
here reads the rendered `figure.layout` rather than the factory's inputs.
"""

import math
from typing import Any, Callable, Dict, Tuple

import pytest
import torch
from dash import dcc

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
def test_no_axis_renders_no_camera_configuration(
    display_kind: str,
    build_display: Callable[..., dcc.Graph],
) -> None:
    """A display given no lock_roll renders no scene configuration at all, so it renders the camera it rendered before this argument existed."""
    display = build_display(lock_roll=None)

    layout = display.figure.layout.to_plotly_json()
    assert "scene" not in layout, (
        "A display given no roll-lock axis must add no scene entry to the rendered "
        f"layout. {display_kind=} {sorted(layout)=}"
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
    display = build_display(lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL)

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
        lock_roll=NON_AXIS_ALIGNED_LOCK_ROLL
    ).figure.layout.scene

    assert free_scene != locked_scene, (
        "Naming a roll-lock axis must change the rendered camera. "
        f"{display_kind=} {free_scene=} {locked_scene=}"
    )
