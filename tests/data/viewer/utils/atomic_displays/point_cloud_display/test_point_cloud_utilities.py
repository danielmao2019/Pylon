"""Tests for point cloud utility functions.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

from dash import html

from data.viewer.utils.atomic_displays.point_cloud_display import (
    point_cloud_to_numpy,
    normalize_point_cloud_id,
    build_point_cloud_id,
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
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 3
    assert stats['dimensions'] == 3
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'center' in stats


def test_get_point_cloud_display_stats_basic_extended():
    """Test basic point cloud statistics with additional checks."""
    points = torch.randn(100, 3, dtype=torch.float32)
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['dimensions'] == 3
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'center' in stats


def test_get_point_cloud_display_stats_with_change_map():
    """Test point cloud statistics with change map."""
    points = torch.randn(100, 3, dtype=torch.float32)
    change_map = torch.randint(0, 3, (100,), dtype=torch.long)
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict, change_map=change_map)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['dimensions'] == 3
    assert 'class_distribution' in stats
    assert isinstance(stats['class_distribution'], dict)


def test_get_point_cloud_display_stats_invalid_points_shape():
    """Test that function works with points having >3 dimensions."""
    points = torch.randn(100, 4, dtype=torch.float32)  # 4D points should work
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['dimensions'] == 4
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'center' in stats


# ================================================================================
# Integration Tests for Utility Functions
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


def test_point_cloud_utility_pipeline(point_cloud_3d):
    """Test complete point cloud utility pipeline."""
    # Convert to numpy
    numpy_pc = point_cloud_to_numpy(point_cloud_3d)
    assert isinstance(numpy_pc, np.ndarray)
    assert numpy_pc.shape == (1000, 3)
    
    # Get statistics
    pc_dict = {'pos': point_cloud_3d}
    stats = get_point_cloud_display_stats(pc_dict)
    assert isinstance(stats, dict)
    assert stats['total_points'] == 1000
    assert stats['dimensions'] == 3
