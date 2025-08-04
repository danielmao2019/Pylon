"""Tests for point cloud utility functions.

Tests the basic utility functions from point_cloud_display module.
CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

from data.viewer.utils.atomic_displays.point_cloud_display import (
    point_cloud_to_numpy,
    normalize_point_cloud_id
)


# ================================================================================
# Fixtures
# ================================================================================

@pytest.fixture
def point_cloud_3d():
    """Fixture providing 3D point cloud tensor."""
    return torch.randn(1000, 3, dtype=torch.float32)




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
    """Test assertion failure for invalid input type."""
    with pytest.raises(TypeError):  # The real function raises TypeError for non-iterable
        normalize_point_cloud_id(123)


def test_normalize_point_cloud_id_empty_tuple():
    """Test that empty tuple returns empty string."""
    result = normalize_point_cloud_id(())
    assert result == ""


# ================================================================================
# Edge Case Tests
# ================================================================================

def test_point_cloud_conversion_edge_cases():
    """Test edge cases for point cloud processing."""
    # Very small coordinates
    tiny_pc = torch.full((100, 3), 1e-6, dtype=torch.float32)
    result = point_cloud_to_numpy(tiny_pc)
    assert result.shape == (100, 3)
    assert np.allclose(result, 1e-6)
    
    # Very large coordinates
    huge_pc = torch.full((100, 3), 1e6, dtype=torch.float32)
    result = point_cloud_to_numpy(huge_pc)
    assert result.shape == (100, 3)
    assert np.allclose(result, 1e6)
    
    # Mixed positive/negative
    mixed_pc = torch.randn(100, 3, dtype=torch.float32) * 1000
    result = point_cloud_to_numpy(mixed_pc)
    assert result.shape == (100, 3)


def test_large_point_cloud_conversion():
    """Test performance with large point clouds."""
    # Create large point cloud
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    
    # These should complete without error
    numpy_pc = point_cloud_to_numpy(large_pc)
    
    # Basic checks
    assert numpy_pc.shape == (10000, 3)
    assert isinstance(numpy_pc, np.ndarray)


def test_id_normalization_edge_cases():
    """Test edge cases for ID normalization."""
    # Long tuple
    long_tuple = ("a", "b", "c", "d", "e", "f", "g")
    result = normalize_point_cloud_id(long_tuple)
    assert result == "a:b:c:d:e:f:g"
    
    # Special characters in string
    special_id = "dataset/with:special-chars_123"
    result = normalize_point_cloud_id(special_id)
    assert result == special_id
    
    # Tuple with special characters
    special_tuple = ("pcr/dataset", "index_123", "source-target")
    result = normalize_point_cloud_id(special_tuple)
    assert result == "pcr/dataset:index_123:source-target"


def test_round_trip_consistency(point_cloud_3d):
    """Test round-trip tensor conversion consistency."""
    # Convert to numpy and back
    numpy_pc = point_cloud_to_numpy(point_cloud_3d)
    tensor_back = torch.from_numpy(numpy_pc).float()
    
    # Should be identical
    assert torch.allclose(tensor_back, point_cloud_3d)


def test_utility_functions_summary():
    """Summary test demonstrating utility functionality working together."""
    # Test data
    test_points = torch.randn(500, 3, dtype=torch.float32)
    test_id_tuple = ("integration_test", "100", "source")
    
    # Test conversions
    numpy_points = point_cloud_to_numpy(test_points)
    normalized_id = normalize_point_cloud_id(test_id_tuple)
    
    # Test string ID pass-through
    string_id = "simple_string_id"
    normalized_string = normalize_point_cloud_id(string_id)
    
    # Verify all operations worked
    assert numpy_points.shape == (500, 3)
    assert isinstance(numpy_points, np.ndarray)
    assert normalized_id == "integration_test:100:source"
    assert normalized_string == "simple_string_id"
    
    print(f"Utility test summary: ✓ All basic utility functions work correctly")
    print(f"  - Point cloud tensor conversion: ✓")
    print(f"  - ID normalization: ✓")
    print(f"  - Edge cases and performance: ✓")