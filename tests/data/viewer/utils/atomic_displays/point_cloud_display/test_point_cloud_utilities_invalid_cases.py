"""Tests for point cloud display utility functions - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from dash import html

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
        get_point_cloud_display_stats("not_a_dict")
    
    assert "Expected dict" in str(exc_info.value)


def test_get_point_cloud_display_stats_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    pc_1d = torch.randn(100, dtype=torch.float32)
    pc_dict_1d = {'pos': pc_1d}
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict_1d)
    assert "Expected 2D tensor [N,D]" in str(exc_info.value)
    
    # 3D tensor
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    pc_dict_3d = {'pos': pc_3d}
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict_3d)
    assert "Expected 2D tensor [N,D]" in str(exc_info.value)
    
    # 4D tensor
    pc_4d = torch.randn(1, 1, 100, 3, dtype=torch.float32)
    pc_dict_4d = {'pos': pc_4d}
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict_4d)
    assert "Expected 2D tensor [N,D]" in str(exc_info.value)


def test_get_point_cloud_display_stats_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 2 channels
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    pc_dict_2ch = {'pos': pc_2ch}
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict_2ch)
    assert "Expected at least 3 coordinates" in str(exc_info.value)
    
    # 1 channel
    pc_1ch = torch.randn(100, 1, dtype=torch.float32)
    pc_dict_1ch = {'pos': pc_1ch}
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict_1ch)
    assert "Expected at least 3 coordinates" in str(exc_info.value)


def test_get_point_cloud_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    pc_dict_empty = {'pos': empty_pc}
    
    with pytest.raises(AssertionError) as exc_info:
        get_point_cloud_display_stats(pc_dict_empty)
    
    assert "Point cloud cannot be empty" in str(exc_info.value)


# ================================================================================
# build_point_cloud_id Tests - Invalid Cases
# ================================================================================

def test_build_point_cloud_id_invalid_datapoint_type():
    """Test assertion failure for invalid datapoint type."""
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id("not_a_dict", "source")
    
    assert "datapoint must be dict" in str(exc_info.value)


def test_build_point_cloud_id_invalid_component_type():
    """Test assertion failure for invalid component type."""
    datapoint = {"meta_info": {"idx": 42}}
    with pytest.raises(AssertionError) as exc_info:
        build_point_cloud_id(datapoint, 123)
    assert "component must be str" in str(exc_info.value)


# ================================================================================
# apply_lod_to_point_cloud Tests - Invalid Cases
# ================================================================================

def test_apply_lod_to_point_cloud_invalid_point_cloud_type():
    """Test assertion failure for invalid point cloud type."""
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points="not_a_tensor",
            camera_state={"eye": {"x": 1, "y": 1, "z": 1}}
        )
    
    assert "points must be torch.Tensor" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_camera_state_type():
    """Test assertion failure for invalid camera state type."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    with pytest.raises(AttributeError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            camera_state="not_a_dict",
            lod_type="continuous"  # This will trigger camera_state validation
        )
    
    assert "'str' object has no attribute 'get'" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_point_cloud_dimensions():
    """Test assertion failure for wrong point cloud dimensions."""
    # 1D tensor
    pc_1d = torch.randn(100, dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc_1d,
            camera_state=camera_state
        )
    assert "points must be (N, 3)" in str(exc_info.value)
    
    # 3D tensor
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc_3d,
            camera_state=camera_state
        )
    assert "points must be (N, 3)" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_point_cloud_channels():
    """Test assertion failure for wrong number of channels."""
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    # 2 channels
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc_2ch,
            camera_state=camera_state
        )
    assert "points must be (N, 3)" in str(exc_info.value)
    
    # 4 channels
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc_4ch,
            camera_state=camera_state
        )
    assert "points must be (N, 3)" in str(exc_info.value)


def test_apply_lod_to_point_cloud_empty_point_cloud():
    """Test assertion failure for empty point cloud."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    with pytest.raises(RuntimeError) as exc_info:
        apply_lod_to_point_cloud(
            points=empty_pc,
            camera_state=camera_state,
            lod_type="continuous"
        )
    
    assert "quantile() input tensor must be non-empty" in str(exc_info.value)



def test_apply_lod_to_point_cloud_invalid_camera_eye_type():
    """Test assertion failure for camera eye not being dict."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    camera_state = {"eye": "not_a_dict"}
    
    with pytest.raises(TypeError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            camera_state=camera_state,
            lod_type="continuous"
        )
    
    assert "string indices must be integers" in str(exc_info.value)


def test_apply_lod_to_point_cloud_missing_camera_eye_coordinates():
    """Test assertion failure for camera eye missing x, y, z coordinates."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    # Missing 'x'
    camera_state = {"eye": {"y": 1, "z": 1}}
    with pytest.raises(KeyError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            camera_state=camera_state,
            lod_type="continuous"
        )
    assert "'x'" in str(exc_info.value)
    
    # Missing 'y'
    camera_state = {"eye": {"x": 1, "z": 1}}
    with pytest.raises(KeyError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            camera_state=camera_state,
            lod_type="continuous"
        )
    assert "'y'" in str(exc_info.value)
    
    # Missing 'z'
    camera_state = {"eye": {"x": 1, "y": 1}}
    with pytest.raises(KeyError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            camera_state=camera_state,
            lod_type="continuous"
        )
    assert "'z'" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_density_percentage_type():
    """Test assertion failure for invalid density_percentage type."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    with pytest.raises(TypeError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            lod_type="none",
            density_percentage="not_int",
            point_cloud_id="test"
        )
    
    assert "'>=' not supported between instances of 'str' and 'int'" in str(exc_info.value)


def test_apply_lod_to_point_cloud_invalid_density_percentage_value():
    """Test assertion failure for invalid density_percentage value."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    # Negative density_percentage
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            lod_type="none",
            density_percentage=-10,
            point_cloud_id="test"
        )
    
    assert "density_percentage must be 1-100" in str(exc_info.value)
    
    # Zero density_percentage
    with pytest.raises(AssertionError) as exc_info:
        apply_lod_to_point_cloud(
            points=pc,
            lod_type="none",
            density_percentage=0,
            point_cloud_id="test"
        )
    
    assert "density_percentage must be 1-100" in str(exc_info.value)


# ================================================================================
# normalize_point_cloud_id Tests - Invalid Cases
# ================================================================================

def test_normalize_point_cloud_id_proper_usage():
    """Test normalize function with proper string and tuple inputs."""
    # String input
    result_str = normalize_point_cloud_id("test_id")
    assert result_str == "test_id"
    
    # Tuple input (proper usage)
    result_tuple = normalize_point_cloud_id(("dataset", 42, "source"))
    assert result_tuple == "dataset:42:source"


def test_normalize_point_cloud_id_empty_string():
    """Test empty string handling."""
    # The function handles empty strings by returning them as-is
    result = normalize_point_cloud_id("")
    assert result == ""


# ================================================================================
# point_cloud_to_numpy Tests - Invalid Cases  
# ================================================================================

def test_point_cloud_to_numpy_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        point_cloud_to_numpy("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_point_cloud_to_numpy_various_dimensions():
    """Test that function works with various tensor dimensions."""
    # 1D tensor (should work)
    pc_1d = torch.randn(100, dtype=torch.float32)
    result_1d = point_cloud_to_numpy(pc_1d)
    assert isinstance(result_1d, np.ndarray)
    assert result_1d.shape == (100,)
    
    # 3D tensor (should work)
    pc_3d = torch.randn(1, 100, 3, dtype=torch.float32)
    result_3d = point_cloud_to_numpy(pc_3d)
    assert isinstance(result_3d, np.ndarray)
    assert result_3d.shape == (1, 100, 3)


def test_point_cloud_to_numpy_various_channels():
    """Test that function works with various channel numbers."""
    # 2 channels (should work)
    pc_2ch = torch.randn(100, 2, dtype=torch.float32)
    result_2ch = point_cloud_to_numpy(pc_2ch)
    assert isinstance(result_2ch, np.ndarray)
    assert result_2ch.shape == (100, 2)
    
    # 4 channels (should work)
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    result_4ch = point_cloud_to_numpy(pc_4ch)
    assert isinstance(result_4ch, np.ndarray)
    assert result_4ch.shape == (100, 4)


def test_point_cloud_to_numpy_empty_tensor():
    """Test that function works with empty tensor."""
    empty_pc = torch.empty((0, 3), dtype=torch.float32)
    
    result = point_cloud_to_numpy(empty_pc)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 3)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_point_cloud_utilities_with_different_dtypes():
    """Test that point cloud utilities work with various dtypes (should work)."""
    # Float64
    pc_f64 = torch.randn(100, 3, dtype=torch.float64)
    stats_f64 = get_point_cloud_display_stats(pc_f64)
    assert isinstance(stats_f64, dict)
    assert stats_f64['total_points'] == 100
    assert stats_f64['dimensions'] == 3
    
    datapoint_f64 = {"meta_info": {"idx": 42}}
    pc_id_f64 = build_point_cloud_id(datapoint_f64, "source")
    assert isinstance(pc_id_f64, tuple)
    
    pc_numpy_f64 = point_cloud_to_numpy(pc_f64)
    assert isinstance(pc_numpy_f64, np.ndarray)
    
    # Integer (unusual but should work)
    pc_int = torch.randint(-10, 10, (100, 3), dtype=torch.int32)
    stats_int = get_point_cloud_display_stats(pc_int)
    assert isinstance(stats_int, dict)
    assert stats_int['total_points'] == 100
    assert stats_int['dimensions'] == 3
    
    datapoint_int = {"meta_info": {"idx": 43}}
    pc_id_int = build_point_cloud_id(datapoint_int, "target")
    assert isinstance(pc_id_int, tuple)
    
    pc_numpy_int = point_cloud_to_numpy(pc_int)
    assert isinstance(pc_numpy_int, np.ndarray)


def test_point_cloud_utilities_extreme_shapes():
    """Test utilities with extreme but valid point cloud shapes."""
    # Single point
    single_point = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    
    stats_single = get_point_cloud_display_stats(single_point)
    assert isinstance(stats_single, dict)
    assert stats_single['total_points'] == 1
    assert stats_single['dimensions'] == 3
    
    datapoint_single = {"meta_info": {"idx": 1}}
    pc_id_single = build_point_cloud_id(datapoint_single, "single")
    assert isinstance(pc_id_single, tuple)
    
    pc_numpy_single = point_cloud_to_numpy(single_point)
    assert pc_numpy_single.shape == (1, 3)
    
    # Very large point cloud (should work but might be slow)
    # Using smaller size for test performance
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    
    stats_large = get_point_cloud_display_stats(large_pc)
    assert isinstance(stats_large, dict)
    assert stats_large['total_points'] == 10000
    assert stats_large['dimensions'] == 3
    
    datapoint_large = {"meta_info": {"idx": 100}}
    pc_id_large = build_point_cloud_id(datapoint_large, "large")
    assert isinstance(pc_id_large, tuple)


def test_apply_lod_to_point_cloud_edge_cases():
    """Test LOD application with edge case inputs."""
    pc = torch.randn(100, 3, dtype=torch.float32)
    
    # High density percentage (should keep most points)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    lod_points, lod_colors, lod_labels = apply_lod_to_point_cloud(
        points=pc,
        lod_type="none",
        density_percentage=90,
        point_cloud_id="test_high"
    )
    assert isinstance(lod_points, torch.Tensor)
    assert lod_points.shape[0] <= 100  # Should not exceed original size
    
    # Low density percentage (minimum valid value)
    lod_points_min, lod_colors_min, lod_labels_min = apply_lod_to_point_cloud(
        points=pc,
        lod_type="none",
        density_percentage=1,
        point_cloud_id="test_low"
    )
    assert isinstance(lod_points_min, torch.Tensor)
    assert lod_points_min.shape[0] <= pc.shape[0]


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
    assert isinstance(stats_large, dict)
    assert stats_large['total_points'] == 100
    assert stats_large['dimensions'] == 3
    
    pc_numpy_large = point_cloud_to_numpy(large_coords)
    assert isinstance(pc_numpy_large, np.ndarray)
    
    # Very small coordinates
    small_coords = torch.full((100, 3), 1e-6, dtype=torch.float32)
    
    stats_small = get_point_cloud_display_stats(small_coords)
    assert isinstance(stats_small, dict)
    assert stats_small['total_points'] == 100
    assert stats_small['dimensions'] == 3
    
    pc_numpy_small = point_cloud_to_numpy(small_coords)
    assert isinstance(pc_numpy_small, np.ndarray)
    
    # Mixed extreme values
    mixed_coords = torch.randn(100, 3, dtype=torch.float32) * 1e6
    
    stats_mixed = get_point_cloud_display_stats(mixed_coords)
    assert isinstance(stats_mixed, dict)
    assert stats_mixed['total_points'] == 100
    assert stats_mixed['dimensions'] == 3
    
    pc_numpy_mixed = point_cloud_to_numpy(mixed_coords)
    assert isinstance(pc_numpy_mixed, np.ndarray)
