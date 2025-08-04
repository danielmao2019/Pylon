"""Tests for point cloud display functionality.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

import plotly.graph_objects as go
from dash import html

from data.viewer.utils.atomic_displays.point_cloud_display import (
    point_cloud_to_numpy,
    normalize_point_cloud_id,
    build_point_cloud_id,
    create_point_cloud_display,
    get_point_cloud_display_stats
)


# ================================================================================
# Fixtures
# ================================================================================

@pytest.fixture
def point_cloud_3d():
    """Fixture providing 3D point cloud tensor."""
    return torch.randn(1000, 3, dtype=torch.float32)


@pytest.fixture
def point_cloud_colors():
    """Fixture providing point cloud colors."""
    return torch.randint(0, 255, (1000, 3), dtype=torch.uint8)


@pytest.fixture
def point_cloud_labels():
    """Fixture providing point cloud labels."""
    return torch.randint(0, 5, (1000,), dtype=torch.long)


@pytest.fixture
def sample_datapoint():
    """Fixture providing sample datapoint for ID building."""
    return {
        'meta_info': {'idx': 42},
        'other_data': 'test'
    }


@pytest.fixture
def setup_viewer_registry():
    """Set up minimal viewer registry for testing."""
    from data.viewer.callbacks import registry
    
    class MinimalBackend:
        def __init__(self):
            self.current_dataset = 'test_dataset'
    
    class MinimalViewer:
        def __init__(self):
            self.backend = MinimalBackend()
    
    # Store original state
    original_viewer = getattr(registry, 'viewer', None)
    
    # Set minimal viewer
    registry.viewer = MinimalViewer()
    
    yield
    
    # Restore original state
    if original_viewer is not None:
        registry.viewer = original_viewer
    elif hasattr(registry, 'viewer'):
        delattr(registry, 'viewer')


# ================================================================================
# point_cloud_to_numpy Tests
# ================================================================================

def test_point_cloud_to_numpy_tensor(point_cloud_3d):
    """Test converting torch tensor to numpy."""
    result = point_cloud_to_numpy(point_cloud_3d)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1000, 3)
    assert np.allclose(result, point_cloud_3d.cpu().numpy())


def test_point_cloud_to_numpy_numpy_passthrough():
    """Test that numpy arrays pass through unchanged."""
    points = np.random.randn(1000, 3).astype(np.float32)
    result = point_cloud_to_numpy(points)
    
    assert isinstance(result, np.ndarray)
    assert result is points
    assert result.shape == (1000, 3)


def test_point_cloud_to_numpy_empty():
    """Test converting empty point cloud."""
    empty_pc = torch.empty(0, 3, dtype=torch.float32)
    result = point_cloud_to_numpy(empty_pc)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 3)


def test_point_cloud_to_numpy_single_point():
    """Test converting single point."""
    single_pc = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    result = point_cloud_to_numpy(single_pc)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)
    assert np.allclose(result, [[1.0, 2.0, 3.0]])


def test_point_cloud_to_numpy_invalid_input():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy("not_a_tensor_or_array")
    
    assert "Expected torch.Tensor or np.ndarray" in str(exc_info.value)


# ================================================================================
# normalize_point_cloud_id Tests
# ================================================================================

def test_normalize_point_cloud_id_string():
    """Test that string IDs pass through unchanged."""
    point_cloud_id = "simple_id"
    result = normalize_point_cloud_id(point_cloud_id)
    
    assert result == "simple_id"


def test_normalize_point_cloud_id_tuple():
    """Test that tuple IDs are converted to colon-separated strings."""
    point_cloud_id = ("pcr/kitti", "42", "source")
    result = normalize_point_cloud_id(point_cloud_id)
    
    assert result == "pcr/kitti:42:source"


@pytest.mark.parametrize("point_cloud_id,expected", [
    ("simple_id", "simple_id"),
    (("pcr/kitti", "42", "source"), "pcr/kitti:42:source"),
    (("change_detection", "10", "union"), "change_detection:10:union"),
    (("single",), "single"),
    (("a", "b", "c", "d"), "a:b:c:d"),
])
def test_normalize_point_cloud_id_various_inputs(point_cloud_id, expected):
    """Test normalize_point_cloud_id with various inputs."""
    result = normalize_point_cloud_id(point_cloud_id)
    assert result == expected


def test_normalize_point_cloud_id_invalid_input():
    """Test TypeError for invalid input type."""
    with pytest.raises(TypeError):
        normalize_point_cloud_id(123)


# ================================================================================
# build_point_cloud_id Tests - Require viewer system setup
# ================================================================================
# NOTE: These tests require the full viewer system to be initialized with registry
# They should be moved to integration tests or run with proper viewer setup

def test_build_point_cloud_id_basic(sample_datapoint, setup_viewer_registry):
    """Test basic point cloud ID building."""
    result = build_point_cloud_id(sample_datapoint, "source")
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "test_dataset"  # From our MinimalBackend
    assert result[1] == 42  # From sample_datapoint meta_info
    assert result[2] == "source"


def test_build_point_cloud_id_different_components(sample_datapoint, setup_viewer_registry):
    """Test building IDs with different component names."""
    components = ["source", "target", "union", "intersection"]
    
    for component in components:
        result = build_point_cloud_id(sample_datapoint, component)
        assert result[2] == component


def test_build_point_cloud_id_invalid_datapoint_type():
    """Test assertion failure for invalid datapoint type."""
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id("not_a_dict", "source")
    
    assert "datapoint must be dict" in str(exc_info.value)


def test_build_point_cloud_id_missing_meta_info(setup_viewer_registry):
    """Test behavior when meta_info is missing (should default to idx=0)."""
    datapoint = {"other_data": "test"}
    
    result = build_point_cloud_id(datapoint, "source")
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)  # dataset name from registry
    assert result[1] == 0  # default idx when meta_info missing
    assert result[2] == "source"


def test_build_point_cloud_id_missing_idx_in_meta_info(setup_viewer_registry):
    """Test behavior when idx is missing from meta_info (should default to 0)."""
    datapoint = {"meta_info": {"other_field": "value"}}
    
    result = build_point_cloud_id(datapoint, "source")
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)  # dataset name from registry
    assert result[1] == 0  # default idx when not in meta_info
    assert result[2] == "source"


def test_build_point_cloud_id_invalid_component_type(sample_datapoint):
    """Test assertion failure for invalid component type."""
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(sample_datapoint, 123)
    
    assert "component must be str" in str(exc_info.value)


# ================================================================================
# create_point_cloud_display Tests
# ================================================================================

def test_create_point_cloud_display_basic(point_cloud_3d):
    """Test basic point cloud display creation."""
    fig = create_point_cloud_display(
        point_cloud_3d, 
        title="Test Point Cloud", 
        lod_type="none"
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Point Cloud"


def test_create_point_cloud_display_with_colors(point_cloud_3d, point_cloud_colors):
    """Test point cloud display with colors."""
    fig = create_point_cloud_display(
        point_cloud_3d,
        title="Colored Point Cloud",
        colors=point_cloud_colors,
        lod_type="none"
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Colored Point Cloud"


def test_create_point_cloud_display_with_labels(point_cloud_3d, point_cloud_labels):
    """Test point cloud display with labels."""
    fig = create_point_cloud_display(
        point_cloud_3d,
        title="Labeled Point Cloud",
        labels=point_cloud_labels,
        lod_type="none"
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Labeled Point Cloud"


def test_create_point_cloud_display_invalid_points_type():
    """Test assertion failure for invalid points type."""
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display("not_a_tensor", "Test")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_create_point_cloud_display_invalid_points_shape():
    """Test assertion failure for invalid points shape."""
    points = torch.randn(100, 4, dtype=torch.float32)  # 4D instead of 3D
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(points, "Test")
    
    assert "Expected 3 coordinates" in str(exc_info.value)


def test_create_point_cloud_display_empty_points():
    """Test assertion failure for empty points."""
    points = torch.empty(0, 3, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(points, "Test")
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


def test_create_point_cloud_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    points = torch.randn(100, 3, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(points, title=123)  # Use keyword arg to avoid positional confusion
    
    assert "Expected str title" in str(exc_info.value)


def test_create_point_cloud_display_with_lod():
    """Test point cloud display with different LOD types."""
    points = torch.randn(1000, 3, dtype=torch.float32)
    camera_state = {'eye': {'x': 1, 'y': 1, 'z': 1}, 'center': {'x': 0, 'y': 0, 'z': 0}, 'up': {'x': 0, 'y': 0, 'z': 1}}
    
    # Test continuous LOD (needs camera_state)
    fig_continuous = create_point_cloud_display(
        points,
        title="Continuous LOD",
        lod_type="continuous",
        camera_state=camera_state
    )
    assert isinstance(fig_continuous, go.Figure)
    
    # Test discrete LOD (needs point_cloud_id and camera_state)
    fig_discrete = create_point_cloud_display(
        points,
        title="Discrete LOD",
        lod_type="discrete",
        point_cloud_id="test_id",
        camera_state=camera_state
    )
    assert isinstance(fig_discrete, go.Figure)
    
    # Test none LOD
    fig_none = create_point_cloud_display(
        points,
        title="No LOD",
        lod_type="none"
    )
    assert isinstance(fig_none, go.Figure)


# ================================================================================
# get_point_cloud_display_stats Tests
# ================================================================================

def test_get_point_cloud_display_stats_basic():
    """Test basic point cloud statistics."""
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0]
    ], dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, html.Ul)
    stats_str = str(stats)
    assert "Total Points: 3" in stats_str
    assert "Dimensions: 3" in stats_str


def test_get_point_cloud_display_stats_basic_extended():
    """Test basic point cloud statistics with additional checks."""
    points = torch.randn(100, 3, dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, html.Ul)
    stats_str = str(stats)
    assert "Total Points: 100" in stats_str
    assert "Dimensions: 3" in stats_str


def test_get_point_cloud_display_stats_with_change_map():
    """Test point cloud statistics with change map."""
    points = torch.randn(100, 3, dtype=torch.float32)
    change_map = torch.randint(0, 3, (100,), dtype=torch.long)
    
    stats = get_point_cloud_display_stats(points, change_map=change_map)
    
    assert isinstance(stats, html.Ul)
    stats_str = str(stats)
    assert "Total Points: 100" in stats_str
    assert "Class Distribution" in stats_str


def test_get_point_cloud_display_stats_invalid_points_type():
    """Test assertion failure for invalid points type."""
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_point_cloud_display_stats_invalid_points_shape():
    """Test that function works with points having >3 dimensions."""
    points = torch.randn(100, 4, dtype=torch.float32)  # 4D points should work
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, html.Ul)
    stats_str = str(stats)
    assert "Total Points: 100" in stats_str
    assert "Dimensions: 4" in stats_str


# ================================================================================
# Integration and Edge Case Tests
# ================================================================================

def test_point_cloud_id_pipeline(sample_datapoint, setup_viewer_registry):
    """Test complete point cloud ID pipeline."""
    # Build ID
    pc_id = build_point_cloud_id(sample_datapoint, "source")
    assert isinstance(pc_id, tuple)
    assert len(pc_id) == 3
    
    # Normalize ID
    normalized = normalize_point_cloud_id(pc_id)
    assert isinstance(normalized, str)
    assert normalized == "test_dataset:42:source"
    
    # Test round-trip consistency
    assert normalize_point_cloud_id(normalized) == normalized


def test_point_cloud_display_pipeline(point_cloud_3d):
    """Test complete point cloud display pipeline."""
    # Convert to numpy
    numpy_pc = point_cloud_to_numpy(point_cloud_3d)
    assert isinstance(numpy_pc, np.ndarray)
    assert numpy_pc.shape == (1000, 3)
    
    # Create display
    fig = create_point_cloud_display(point_cloud_3d, title="Pipeline Test", lod_type="none")
    assert isinstance(fig, go.Figure)
    
    # Get statistics
    stats = get_point_cloud_display_stats(point_cloud_3d)
    assert isinstance(stats, html.Ul)


def test_large_point_cloud_performance():
    """Test performance with large point clouds."""
    # Create large point cloud
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    
    # These should complete without error
    numpy_pc = point_cloud_to_numpy(large_pc)
    fig = create_point_cloud_display(large_pc, title="Large PC Test", lod_type="none")
    stats = get_point_cloud_display_stats(large_pc)
    
    # Basic checks
    assert numpy_pc.shape == (10000, 3)
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, html.Ul)


def test_edge_case_point_clouds():
    """Test edge cases for point cloud processing."""
    # Very small coordinates
    tiny_pc = torch.full((100, 3), 1e-6, dtype=torch.float32)
    fig = create_point_cloud_display(tiny_pc, title="Tiny PC", lod_type="none")
    assert isinstance(fig, go.Figure)
    
    # Very large coordinates
    huge_pc = torch.full((100, 3), 1e6, dtype=torch.float32)
    fig = create_point_cloud_display(huge_pc, title="Huge PC", lod_type="none")
    assert isinstance(fig, go.Figure)
    
    # Mixed positive/negative
    mixed_pc = torch.randn(100, 3, dtype=torch.float32) * 1000
    fig = create_point_cloud_display(mixed_pc, title="Mixed PC", lod_type="none")
    assert isinstance(fig, go.Figure)