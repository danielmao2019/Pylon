"""Tests for the Dash point-cloud style-args functions.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.

Covers the branch's new opt-in `point_size` / `point_color` overrides on the
synchronous Dash point-cloud builders.
"""

import numpy as np
import plotly.graph_objects as go
import pytest
import torch
from dash import dcc

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.viewer.utils.displays.points.dash.core_points_display import (
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_SIZE_FLOOR,
    DEFAULT_POINT_SIZE_RATIO,
    create_dash_points_component,
    create_dash_points_display,
    create_dash_points_scene,
)

# ================================================================================
# Fixtures
# ================================================================================


@pytest.fixture
def large_radius_xyz():
    """Synthetic xyz whose bounding-sphere radius is large enough that the size
    heuristic exceeds the floor.

    The eight cube corners at +-1000 are centered at the origin, so every point
    is at distance ``sqrt(3) * 1000`` from the center.
    """
    corner = 1000.0
    coords = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                coords.append([sx * corner, sy * corner, sz * corner])
    return torch.tensor(coords, dtype=torch.float32)


@pytest.fixture
def tiny_radius_xyz():
    """Synthetic xyz whose bounding-sphere radius is so small the size heuristic
    falls back to the floor."""
    corner = 0.001
    coords = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                coords.append([sx * corner, sy * corner, sz * corner])
    return torch.tensor(coords, dtype=torch.float32)


def _expected_radius(xyz: torch.Tensor) -> float:
    """Reproduce the source's bounding-radius computation for assertions."""
    points_np = xyz.detach().cpu().numpy()
    center = points_np.mean(axis=0)
    return float(np.linalg.norm(points_np - center, axis=1).max())


# ================================================================================
# create_dash_points_scene - point_size behavior
# ================================================================================


def test_scene_explicit_point_size_used(large_radius_xyz):
    """Explicit point_size sets marker.size verbatim."""
    pc = PointCloud(xyz=large_radius_xyz)
    trace = create_dash_points_scene(point_cloud=pc, point_size=7.5)

    assert isinstance(trace, go.Scatter3d)
    assert trace.marker.size == 7.5


def test_scene_point_size_falls_back_to_radius_heuristic(large_radius_xyz):
    """When point_size is None and radius*ratio beats the floor, that value is used."""
    pc = PointCloud(xyz=large_radius_xyz)
    radius = _expected_radius(large_radius_xyz)
    expected = max(DEFAULT_POINT_SIZE_FLOOR, radius * DEFAULT_POINT_SIZE_RATIO)
    # Guard the test's own premise: this branch must exceed the floor.
    assert expected > DEFAULT_POINT_SIZE_FLOOR

    trace = create_dash_points_scene(point_cloud=pc, point_size=None)

    assert trace.marker.size == pytest.approx(expected)
    assert trace.marker.size == pytest.approx(radius * DEFAULT_POINT_SIZE_RATIO)


def test_scene_point_size_falls_back_to_floor(tiny_radius_xyz):
    """When point_size is None and radius*ratio is below the floor, the floor is used."""
    pc = PointCloud(xyz=tiny_radius_xyz)
    radius = _expected_radius(tiny_radius_xyz)
    # Guard the test's own premise: this branch must hit the floor.
    assert radius * DEFAULT_POINT_SIZE_RATIO < DEFAULT_POINT_SIZE_FLOOR

    trace = create_dash_points_scene(point_cloud=pc, point_size=None)

    assert trace.marker.size == pytest.approx(DEFAULT_POINT_SIZE_FLOOR)


# ================================================================================
# create_dash_points_scene - point_color behavior
# ================================================================================


def test_scene_explicit_point_color_uniform(large_radius_xyz):
    """Explicit point_color sets a single uniform marker.color string."""
    rgb = torch.randint(0, 256, (8, 3), dtype=torch.uint8)
    pc = PointCloud(xyz=large_radius_xyz, data={"rgb": rgb})

    trace = create_dash_points_scene(point_cloud=pc, point_color="#ff0000")

    assert trace.marker.color == "#ff0000"


def test_scene_point_color_none_uses_per_point_rgb(large_radius_xyz):
    """When point_color is None and rgb exists, per-point colors are used."""
    rgb = torch.arange(8 * 3, dtype=torch.uint8).reshape(8, 3)
    pc = PointCloud(xyz=large_radius_xyz, data={"rgb": rgb})

    trace = create_dash_points_scene(point_cloud=pc, point_color=None)

    color = trace.marker.color
    assert isinstance(color, np.ndarray)
    assert color.shape == (8, 3)
    np.testing.assert_array_equal(color, rgb.detach().cpu().numpy())


def test_scene_point_color_none_no_rgb_uses_default(large_radius_xyz):
    """When point_color is None and no rgb, DEFAULT_POINT_COLOR is used."""
    pc = PointCloud(xyz=large_radius_xyz)
    assert "rgb" not in pc.field_names()

    trace = create_dash_points_scene(point_cloud=pc, point_color=None)

    assert trace.marker.color == DEFAULT_POINT_COLOR


# ================================================================================
# create_dash_points_display / create_dash_points_component
# ================================================================================


def test_create_dash_points_display_returns_graph(large_radius_xyz):
    """create_dash_points_display returns a dcc.Graph wrapping the scene."""
    pc = PointCloud(xyz=large_radius_xyz)

    graph = create_dash_points_display(point_cloud=pc)

    assert isinstance(graph, dcc.Graph)
    assert isinstance(graph.figure, go.Figure)
    assert len(graph.figure.data) == 1
    assert isinstance(graph.figure.data[0], go.Scatter3d)


def test_create_dash_points_display_passes_style_args(large_radius_xyz):
    """Style overrides reach the wrapped Scatter3d marker."""
    pc = PointCloud(xyz=large_radius_xyz)

    graph = create_dash_points_display(
        point_cloud=pc, point_size=4.0, point_color="#00ff00"
    )

    trace = graph.figure.data[0]
    assert trace.marker.size == 4.0
    assert trace.marker.color == "#00ff00"


def test_create_dash_points_component_wraps_scene(large_radius_xyz):
    """create_dash_points_component wraps a Scatter3d into a single-trace Graph."""
    pc = PointCloud(xyz=large_radius_xyz)
    scene = create_dash_points_scene(point_cloud=pc, point_size=3.0)

    graph = create_dash_points_component(
        scene=scene, controls=lambda *args, **kwargs: None
    )

    assert isinstance(graph, dcc.Graph)
    assert isinstance(graph.figure, go.Figure)
    assert len(graph.figure.data) == 1
    assert graph.figure.data[0].marker.size == 3.0
