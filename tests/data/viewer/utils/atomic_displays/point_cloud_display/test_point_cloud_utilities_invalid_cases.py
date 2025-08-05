"""Tests for point cloud display utility functions - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np

from data.viewer.utils.atomic_displays.point_cloud_display import (
    get_point_cloud_display_stats,
    build_point_cloud_id,
    apply_lod_to_point_cloud,
    normalize_point_cloud_id,
    point_cloud_to_numpy
)


# ================================================================================
# get_point_cloud_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_point_cloud_display_stats_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_point_cloud_display_stats_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    pc_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_1d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)
    
    # 3D tensor
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_3d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)
    
    # 4D tensor
    pc_4d = torch.randn(1, 1, 100, 3, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_4d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)


def test_get_point_cloud_display_stats_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 2 channels
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_2ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)
    
    # 4 channels
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_4ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)
    
    # 1 channel
    pc_1ch = torch.randn(100, 1, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_1ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)


def test_get_point_cloud_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(empty_pc)
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


# ================================================================================
# build_point_cloud_id Tests - Invalid Cases
# ================================================================================

def test_build_point_cloud_id_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_build_point_cloud_id_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    pc_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(pc_1d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)
    
    # 3D tensor
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(pc_3d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)


def test_build_point_cloud_id_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 2 channels
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(pc_2ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)
    
    # 4 channels
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(pc_4ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)


def test_build_point_cloud_id_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(empty_pc)
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


# ================================================================================
# apply_lod_to_point_cloud Tests - Invalid Cases
# ================================================================================

def test_apply_lod_to_point_cloud_invalid_point_cloud_type():
    """Test assertion failure for invalid point cloud type."""
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud("not_a_tensor", {"eye": {"x": 1, "y": 1, "z": 1}})
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_camera_state_type():
    """Test assertion failure for invalid camera state type."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, "not_a_dict")
    
    assert "camera_state must be dict" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_point_cloud_dimensions():
    """Test assertion failure for wrong point cloud dimensions."""
    # 1D tensor
    pc_1d = torch.randn(100, dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc_1d, camera_state)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)
    
    # 3D tensor
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc_3d, camera_state)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_point_cloud_channels():
    """Test assertion failure for wrong number of channels."""
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    # 2 channels
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc_2ch, camera_state)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)
    
    # 4 channels
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc_4ch, camera_state)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)


def test_apply_lod_to_point_cloud_empty_point_cloud():
    """Test assertion failure for empty point cloud."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(empty_pc, camera_state)
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


def test_apply_lod_to_point_cloud_missing_camera_eye():
    """Test assertion failure for camera state missing 'eye' key."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    camera_state = {"center": {"x": 0, "y": 0, "z": 0}}
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state)
    
    assert "camera_state must have 'eye'" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_camera_eye_type():
    """Test assertion failure for camera eye not being dict."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    camera_state = {"eye": "not_a_dict"}
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state)
    
    assert "camera_state['eye'] must be dict" in str(exc_info.value)


def test_apply_lod_to_point_cloud_missing_camera_eye_coordinates():
    """Test assertion failure for camera eye missing x, y, z coordinates."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    # Missing 'x'
    camera_state = {"eye": {"y": 1, "z": 1}}
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state)
    assert "camera_state['eye'] must have 'x'" in str(exc_info.value)
    
    # Missing 'y'
    camera_state = {"eye": {"x": 1, "z": 1}}
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state)
    assert "camera_state['eye'] must have 'y'" in str(exc_info.value)
    
    # Missing 'z'
    camera_state = {"eye": {"x": 1, "y": 1}}
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state)
    assert "camera_state['eye'] must have 'z'" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_max_points_type():
    """Test assertion failure for invalid max_points type."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state, max_points="not_int")
    
    assert "max_points must be int" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_max_points_value():
    """Test assertion failure for invalid max_points value."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    # Negative max_points
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state, max_points=-10)
    
    assert "max_points must be positive" in str(exc_info.value)
    
    # Zero max_points
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(pc, camera_state, max_points=0)
    
    assert "max_points must be positive" in str(exc_info.value)


# ================================================================================
# normalize_point_cloud_id Tests - Invalid Cases
# ================================================================================

def test_normalize_point_cloud_id_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        normalize_point_cloud_id(123)
    
    assert "Expected str point_cloud_id" in str(exc_info.value)


def test_normalize_point_cloud_id_empty_string():
    """Test assertion failure for empty string."""
    with pytest.raises(AssertionError) as exc_info:
        normalize_point_cloud_id("")
    
    assert "point_cloud_id cannot be empty" in str(exc_info.value)


# ================================================================================
# point_cloud_to_numpy Tests - Invalid Cases  
# ================================================================================

def test_point_cloud_to_numpy_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_point_cloud_to_numpy_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    pc_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy(pc_1d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)
    
    # 3D tensor
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy(pc_3d)
    assert "Expected 2D tensor [N, 3]" in str(exc_info.value)


def test_point_cloud_to_numpy_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 2 channels
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy(pc_2ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)
    
    # 4 channels
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy(pc_4ch)
    assert "Expected 3 coordinates [N, 3]" in str(exc_info.value)


def test_point_cloud_to_numpy_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy(empty_pc)
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_point_cloud_utilities_with_different_dtypes():
    """Test that point cloud utilities work with various dtypes (should work)."""
    # Float64
    pc_f64 = torch.randn(100, 3, dtype=torch.float64)
    stats_f64 = get_point_cloud_display_stats(pc_f64)
    assert isinstance(stats_f64, html.Ul)
    
    pc_id_f64 = build_point_cloud_id(pc_f64)
    assert isinstance(pc_id_f64, str)
    
    pc_numpy_f64 = point_cloud_to_numpy(pc_f64)
    assert isinstance(pc_numpy_f64, np.ndarray)
    
    # Integer (unusual but should work)
    pc_int = torch.randint(-10, 10, (100, 3), dtype=torch.int32)
    stats_int = get_point_cloud_display_stats(pc_int)
    assert isinstance(stats_int, html.Ul)
    
    pc_id_int = build_point_cloud_id(pc_int)
    assert isinstance(pc_id_int, str)
    
    pc_numpy_int = point_cloud_to_numpy(pc_int)
    assert isinstance(pc_numpy_int, np.ndarray)


def test_point_cloud_utilities_extreme_shapes():
    """Test utilities with extreme but valid point cloud shapes."""
    # Single point
    single_point = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    
    stats_single = get_point_cloud_display_stats(single_point)
    assert isinstance(stats_single, html.Ul)
    
    pc_id_single = build_point_cloud_id(single_point)
    assert isinstance(pc_id_single, str)
    
    pc_numpy_single = point_cloud_to_numpy(single_point)
    assert pc_numpy_single.shape == (1, 3)
    
    # Very large point cloud (should work but might be slow)
    # Using smaller size for test performance
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    
    stats_large = get_point_cloud_display_stats(large_pc)
    assert isinstance(stats_large, html.Ul)
    
    pc_id_large = build_point_cloud_id(large_pc)
    assert isinstance(pc_id_large, str)


def test_apply_lod_to_point_cloud_edge_cases():
    """Test LOD application with edge case inputs."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    # Very large max_points (larger than point cloud)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    lod_pc = apply_lod_to_point_cloud(pc, camera_state, max_points=10000)
    assert isinstance(lod_pc, torch.Tensor)
    assert lod_pc.shape[0] <= 100  # Should not exceed original size
    
    # max_points = 1 (minimum valid value)
    lod_pc_min = apply_lod_to_point_cloud(pc, camera_state, max_points=1)
    assert isinstance(lod_pc_min, torch.Tensor)
    assert lod_pc_min.shape[0] == 1


def test_normalize_point_cloud_id_edge_cases():
    """Test point cloud ID normalization with edge cases."""
    # Very long string
    long_id = "a" * 1000
    normalized_long = normalize_point_cloud_id(long_id)
    assert isinstance(normalized_long, str)
    assert len(normalized_long) > 0
    
    # String with special characters
    special_id = "point_cloud@#$%^&*()[]{}|\\:;\"'<>?,./"
    normalized_special = normalize_point_cloud_id(special_id)
    assert isinstance(normalized_special, str)
    assert len(normalized_special) > 0
    
    # Single character
    single_char = "a"
    normalized_single = normalize_point_cloud_id(single_char)
    assert isinstance(normalized_single, str)
    assert normalized_single == single_char or len(normalized_single) > 0


def test_point_cloud_utilities_with_extreme_values():
    """Test utilities with extreme coordinate values."""
    # Very large coordinates
    large_coords = torch.full((100, 3), 1e6, dtype=torch.float32)
    
    stats_large = get_point_cloud_display_stats(large_coords)
    assert isinstance(stats_large, html.Ul)
    
    pc_numpy_large = point_cloud_to_numpy(large_coords)
    assert isinstance(pc_numpy_large, np.ndarray)
    
    # Very small coordinates
    small_coords = torch.full((100, 3), 1e-6, dtype=torch.float32)
    
    stats_small = get_point_cloud_display_stats(small_coords)
    assert isinstance(stats_small, html.Ul)
    
    pc_numpy_small = point_cloud_to_numpy(small_coords)
    assert isinstance(pc_numpy_small, np.ndarray)
    
    # Mixed extreme values
    mixed_coords = torch.randn(100, 3, dtype=torch.float32) * 1e6
    
    stats_mixed = get_point_cloud_display_stats(mixed_coords)
    assert isinstance(stats_mixed, html.Ul)
    
    pc_numpy_mixed = point_cloud_to_numpy(mixed_coords)
    assert isinstance(pc_numpy_mixed, np.ndarray)
