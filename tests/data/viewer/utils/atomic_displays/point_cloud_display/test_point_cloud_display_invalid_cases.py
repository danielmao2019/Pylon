"""Tests for point cloud display functionality - Invalid cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.point_cloud_display import (
    create_point_cloud_display,
    build_point_cloud_id,
    normalize_point_cloud_id,
    get_point_cloud_display_stats
)


# ================================================================================
# Fixtures
# ================================================================================

@pytest.fixture
def sample_datapoint():
    """Fixture providing sample datapoint for ID building."""
    return {
        'meta_info': {'idx': 42},
        'other_data': 'test'
    }


# ================================================================================
# create_point_cloud_display Tests - Invalid Cases
# ================================================================================

def test_create_point_cloud_display_invalid_pc_type():
    """Test assertion failure for invalid pc type."""
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(
            pc="not_a_dict",  # Should be a dict
            color_key=None,
            highlight_indices=None,
            title="Test"
        )
    
    assert "Expected dict for pc" in str(exc_info.value)


def test_create_point_cloud_display_missing_pos_key():
    """Test assertion failure for missing 'pos' key."""
    pc = {'rgb': torch.randn(100, 3)}  # Missing 'pos' key
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(
            pc=pc,
            color_key=None,
            highlight_indices=None,
            title="Test"
        )
    
    assert "pc must have 'pos' key" in str(exc_info.value)


def test_create_point_cloud_display_invalid_points_shape():
    """Test assertion failure for invalid points shape."""
    points = torch.randn(100, 4, dtype=torch.float32)  # 4D instead of 3D
    pc = {'pos': points}
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(
            pc=pc,
            color_key=None,
            highlight_indices=None,
            title="Test"
        )
    
    assert "Expected 3 coordinates" in str(exc_info.value)


def test_create_point_cloud_display_empty_points():
    """Test assertion failure for empty points."""
    points = torch.empty(0, 3, dtype=torch.float32)
    pc = {'pos': points}
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(
            pc=pc,
            color_key=None,
            highlight_indices=None,
            title="Test"
        )
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


def test_create_point_cloud_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    points = torch.randn(100, 3, dtype=torch.float32)
    pc = {'pos': points}
    
    with pytest.raises(AssertionError) as exc_info:
        create_point_cloud_display(pc=pc, color_key=None, title=123)  # Invalid title type
    
    assert "Expected str title" in str(exc_info.value)


# ================================================================================
# build_point_cloud_id Tests - Invalid Cases
# ================================================================================

def test_build_point_cloud_id_invalid_datapoint_type():
    """Test assertion failure for invalid datapoint type."""
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id("not_a_dict", "source")
    
    assert "datapoint must be dict" in str(exc_info.value)


def test_build_point_cloud_id_invalid_component_type(sample_datapoint):
    """Test assertion failure for invalid component type."""
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(sample_datapoint, 123)
    
    assert "component must be str" in str(exc_info.value)


# ================================================================================
# normalize_point_cloud_id Tests - Invalid Cases
# ================================================================================

def test_normalize_point_cloud_id_invalid_input():
    """Test TypeError for invalid input type."""
    with pytest.raises(TypeError):
        normalize_point_cloud_id(123)


# ================================================================================
# get_point_cloud_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_point_cloud_display_stats_invalid_pc_dict_type():
    """Test assertion failure for invalid pc_dict type."""
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats("not_a_dict")
    
    assert "Expected dict" in str(exc_info.value)


def test_get_point_cloud_display_stats_missing_pos_key():
    """Test assertion failure for missing 'pos' key."""
    pc_dict = {'rgb': torch.randn(100, 3)}  # Missing 'pos'
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict)
    
    assert "pc_dict must have 'pos' key" in str(exc_info.value)


def test_get_point_cloud_display_stats_invalid_points_type():
    """Test assertion failure for invalid points tensor type."""
    pc_dict = {'pos': "not_a_tensor"}  # pos is not a tensor
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict)
    
    assert "Expected torch.Tensor" in str(exc_info.value)
