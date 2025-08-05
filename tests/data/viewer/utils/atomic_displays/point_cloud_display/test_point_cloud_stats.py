"""Tests for point cloud display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any
from dash import html

from data.viewer.utils.atomic_displays.point_cloud_display import (
    get_point_cloud_display_stats,
    build_point_cloud_id,
    apply_lod_to_point_cloud,
    normalize_point_cloud_id,
    point_cloud_to_numpy
)


# ================================================================================
# get_point_cloud_display_stats Tests - Valid Cases
# ================================================================================

def test_get_point_cloud_display_stats_basic(point_cloud_3d):
    """Test basic point cloud statistics calculation."""
    stats = get_point_cloud_display_stats(point_cloud_3d)
    
    assert isinstance(stats, html.Ul)
    assert len(stats.children) >= 6  # At least 6 basic stats items
    
    # Check that stats contain expected text patterns
    stats_text = str(stats)
    assert "Total Points: 1000" in stats_text
    assert "Dimensions: 3" in stats_text
    assert "X Range:" in stats_text
    assert "Y Range:" in stats_text
    assert "Z Range:" in stats_text
    assert "Center:" in stats_text


def test_get_point_cloud_display_stats_known_values():
    """Test statistics with known point cloud values."""
    # Create point cloud with known properties
    points = torch.zeros(100, 3, dtype=torch.float32)
    
    # Set specific coordinate ranges
    points[:25, 0] = 1.0    # X coordinates: 25 points at x=1
    points[25:50, 0] = 2.0  # X coordinates: 25 points at x=2
    points[50:75, 1] = 3.0  # Y coordinates: 25 points at y=3
    points[75:, 2] = 4.0    # Z coordinates: 25 points at z=4
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, html.Ul)
    
    # Check that stats contain expected text patterns
    stats_text = str(stats)
    assert "Total Points: 100" in stats_text
    assert "[0.00, 2.00]" in stats_text  # X range
    assert "[0.00, 3.00]" in stats_text  # Y range
    assert "[0.00, 4.00]" in stats_text  # Z range


def test_get_point_cloud_display_stats_single_point():
    """Test statistics with single point."""
    single_point = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(single_point)
    
    assert isinstance(stats, html.Ul)
    
    # Check that stats contain expected text patterns
    stats_text = str(stats)
    assert "Total Points: 1" in stats_text
    assert "[1.50, 1.50]" in stats_text  # X range
    assert "[2.50, 2.50]" in stats_text  # Y range  
    assert "[3.50, 3.50]" in stats_text  # Z range


def test_get_point_cloud_display_stats_uniform_distribution():
    """Test statistics with uniformly distributed points."""
    # Create points in unit cube [0,1]^3
    points = torch.rand(1000, 3, dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, html.Ul)
    
    # Check that stats contain expected text patterns
    stats_text = str(stats)
    assert "Total Points: 1000" in stats_text
    assert "Dimensions: 3" in stats_text
    # Ranges should be approximately [0, 1] for each dimension (allowing some randomness tolerance)
    assert "X Range:" in stats_text
    assert "Y Range:" in stats_text
    assert "Z Range:" in stats_text


@pytest.mark.parametrize("n_points", [10, 100, 1000, 5000])
def test_get_point_cloud_display_stats_various_sizes(n_points):
    """Test statistics with various point cloud sizes."""
    points = torch.randn(n_points, 3, dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, html.Ul)
    
    # Check that stats contain expected text patterns
    stats_text = str(stats)
    assert f"Total Points: {n_points}" in stats_text


def test_get_point_cloud_display_stats_different_dtypes():
    """Test statistics with different tensor dtypes."""
    # Float32
    points_f32 = torch.randn(100, 3, dtype=torch.float32)
    stats_f32 = get_point_cloud_display_stats(points_f32)
    assert isinstance(stats_f32, html.Ul)
    
    # Float64
    points_f64 = torch.randn(100, 3, dtype=torch.float64)
    stats_f64 = get_point_cloud_display_stats(points_f64)
    assert isinstance(stats_f64, html.Ul)
    
    # Integer (unusual but should work)
    points_int = torch.randint(-10, 10, (100, 3), dtype=torch.int32)
    stats_int = get_point_cloud_display_stats(points_int)
    assert isinstance(stats_int, html.Ul)


def test_get_point_cloud_display_stats_extreme_coordinates():
    """Test statistics with extreme coordinate values."""
    # Very large coordinates
    large_points = torch.full((100, 3), 1e6, dtype=torch.float32)
    stats_large = get_point_cloud_display_stats(large_points)
    assert isinstance(stats_large, html.Ul)
    stats_text = str(stats_large)
    assert "Total Points: 100" in stats_text
    
    # Very small coordinates
    small_points = torch.full((100, 3), 1e-6, dtype=torch.float32)
    stats_small = get_point_cloud_display_stats(small_points)
    assert isinstance(stats_small, html.Ul)
    
    # Mixed positive/negative
    mixed_points = torch.randn(100, 3, dtype=torch.float32) * 1000
    stats_mixed = get_point_cloud_display_stats(mixed_points)
    assert isinstance(stats_mixed, html.Ul)


# ================================================================================
# build_point_cloud_id Tests - Valid Cases
# ================================================================================

def test_build_point_cloud_id_basic():
    """Test basic point cloud ID generation without initialized registry."""
    datapoint = {"meta_info": {"idx": 42}}
    component = "source"
    
    pc_id = build_point_cloud_id(datapoint, component)
    
    assert isinstance(pc_id, tuple)
    assert len(pc_id) == 3
    assert pc_id[0] == "unknown"  # Expected when registry.viewer.backend not available
    assert pc_id[1] == 42
    assert pc_id[2] == "source"


def test_build_point_cloud_id_deterministic():
    """Test that point cloud ID generation is deterministic."""
    datapoint = {"meta_info": {"idx": 10}}
    component = "target"
    
    pc_id1 = build_point_cloud_id(datapoint, component)
    pc_id2 = build_point_cloud_id(datapoint, component)
    
    assert isinstance(pc_id1, tuple)
    assert isinstance(pc_id2, tuple)
    assert pc_id1 == pc_id2  # Should be identical for same input


def test_build_point_cloud_id_different_inputs():
    """Test that different point clouds generate different IDs."""
    datapoint1 = {"meta_info": {"idx": 42}}
    datapoint2 = {"meta_info": {"idx": 123}}
    
    id1 = build_point_cloud_id(datapoint1, "source")
    id2 = build_point_cloud_id(datapoint2, "source")
    
    assert isinstance(id1, tuple)
    assert isinstance(id2, tuple)
    assert len(id1) == 3
    assert len(id2) == 3
    assert id1 != id2  # Different inputs should generate different IDs


def test_build_point_cloud_id_various_sizes():
    """Test point cloud ID generation with various datapoint indices."""
    indices = [1, 10, 100, 1000]
    ids = []
    
    for idx in indices:
        datapoint = {"meta_info": {"idx": idx}}
        pc_id = build_point_cloud_id(datapoint, "source")
        
        assert isinstance(pc_id, tuple)
        assert len(pc_id) == 3
        assert pc_id[1] == idx  # Check the index is correct
        ids.append(pc_id)
    
    # All IDs should be different
    assert len(set(ids)) == len(ids)


def test_build_point_cloud_id_single_point():
    """Test point cloud ID generation with single datapoint."""
    datapoint = {"meta_info": {"idx": 1}}
    pc_id = build_point_cloud_id(datapoint, "target")
    
    assert isinstance(pc_id, tuple)
    assert len(pc_id) == 3
    assert pc_id[1] == 1
    assert pc_id[2] == "target"


def test_build_point_cloud_id_different_components():
    """Test point cloud ID generation with different components."""
    datapoint = {"meta_info": {"idx": 100}}
    
    # Different components
    id_source = build_point_cloud_id(datapoint, "source")
    id_target = build_point_cloud_id(datapoint, "target")
    id_change = build_point_cloud_id(datapoint, "change_map")
    
    assert isinstance(id_source, tuple)
    assert isinstance(id_target, tuple)
    assert isinstance(id_change, tuple)
    
    # Same datapoint index, different components
    assert id_source[1] == id_target[1] == id_change[1] == 100
    assert id_source[2] == "source"
    assert id_target[2] == "target"
    assert id_change[2] == "change_map"
    
    # All IDs should be different
    assert id_source != id_target != id_change


# ================================================================================
# apply_lod_to_point_cloud Tests - Valid Cases
# ================================================================================

def test_apply_lod_to_point_cloud_basic(point_cloud_3d, camera_state):
    """Test basic LOD application."""
    points, colors, labels = apply_lod_to_point_cloud(
        points=point_cloud_3d,
        camera_state=camera_state,
        lod_type="none",
        density_percentage=50,
        point_cloud_id="test_pc_basic"
    )
    
    assert isinstance(points, torch.Tensor)
    assert points.shape[1] == 3  # Should maintain 3D coordinates
    assert points.shape[0] <= point_cloud_3d.shape[0]  # Should not exceed original
    assert colors is None  # No colors provided
    assert labels is None  # No labels provided


def test_apply_lod_to_point_cloud_no_reduction_needed(camera_state):
    """Test LOD when no reduction is needed."""
    small_pc = torch.randn(50, 3, dtype=torch.float32)
    points, colors, labels = apply_lod_to_point_cloud(
        points=small_pc,
        lod_type="none",
        density_percentage=100  # No reduction
    )
    
    # Should return original point cloud since no reduction needed
    assert isinstance(points, torch.Tensor)
    assert points.shape[0] == 50  # Should keep all points
    assert points.shape[1] == 3
    assert torch.allclose(points, small_pc)  # Should be identical


def test_apply_lod_to_point_cloud_extreme_reduction(point_cloud_3d, camera_state):
    """Test LOD with extreme reduction (1% of points)."""
    points, colors, labels = apply_lod_to_point_cloud(
        points=point_cloud_3d,
        camera_state=camera_state,
        lod_type="none",
        density_percentage=1,  # Only 1% of points
        point_cloud_id="test_pc_extreme"
    )
    
    assert isinstance(points, torch.Tensor)
    assert points.shape[1] == 3
    assert points.shape[0] <= point_cloud_3d.shape[0] * 0.02  # Should be heavily reduced


def test_apply_lod_to_point_cloud_various_density_percentages(point_cloud_3d, camera_state):
    """Test LOD with various density percentage values."""
    density_values = [10, 25, 50, 75, 90]
    
    for density_pct in density_values:
        points, colors, labels = apply_lod_to_point_cloud(
            points=point_cloud_3d,
            camera_state=camera_state,
            lod_type="none",
            density_percentage=density_pct,
            point_cloud_id=f"test_pc_{density_pct}"
        )
        
        assert isinstance(points, torch.Tensor)
        assert points.shape[1] == 3
        assert points.shape[0] <= point_cloud_3d.shape[0]


def test_apply_lod_to_point_cloud_different_camera_positions():
    """Test LOD with different camera positions using continuous LOD."""
    pc = torch.randn(1000, 3, dtype=torch.float32) * 10.0  # Spread out points
    
    # Camera at origin
    camera_origin = {"eye": {"x": 0, "y": 0, "z": 0}}
    points_origin, colors_origin, labels_origin = apply_lod_to_point_cloud(
        points=pc,
        camera_state=camera_origin,
        lod_type="continuous"
    )
    
    # Camera far away
    camera_far = {"eye": {"x": 100, "y": 100, "z": 100}}
    points_far, colors_far, labels_far = apply_lod_to_point_cloud(
        points=pc,
        camera_state=camera_far,
        lod_type="continuous"
    )
    
    # Both should return valid results
    assert isinstance(points_origin, torch.Tensor)
    assert isinstance(points_far, torch.Tensor)
    assert points_origin.shape[1] == 3
    assert points_far.shape[1] == 3


def test_apply_lod_to_point_cloud_single_point(camera_state):
    """Test LOD with single point."""
    single_point = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    points, colors, labels = apply_lod_to_point_cloud(
        points=single_point,
        lod_type="none",
        density_percentage=100  # Keep all points
    )
    
    assert isinstance(points, torch.Tensor)
    assert torch.allclose(points, single_point)  # Should return the same point


# ================================================================================
# normalize_point_cloud_id Tests - Valid Cases
# ================================================================================

def test_normalize_point_cloud_id_basic():
    """Test basic point cloud ID normalization."""
    original_id = "my_point_cloud_123"
    normalized_id = normalize_point_cloud_id(original_id)
    
    assert isinstance(normalized_id, str)
    assert len(normalized_id) > 0


def test_normalize_point_cloud_id_various_inputs():
    """Test normalization with various input strings."""
    test_ids = [
        "simple_id",
        "point_cloud_with_numbers_123",
        "UPPERCASE_ID",
        "Mixed_Case_ID_456",
        "id-with-dashes",
        "id.with.dots",
        "id_with_underscores_789"
    ]
    
    for original_id in test_ids:
        normalized_id = normalize_point_cloud_id(original_id)
        
        assert isinstance(normalized_id, str)
        assert len(normalized_id) > 0


def test_normalize_point_cloud_id_special_characters():
    """Test normalization with special characters."""
    special_id = "point@cloud#with$special%chars!"
    normalized_id = normalize_point_cloud_id(special_id)
    
    assert isinstance(normalized_id, str)
    assert len(normalized_id) > 0


def test_normalize_point_cloud_id_long_string():
    """Test normalization with very long string."""
    long_id = "very_long_point_cloud_identifier_" * 10  # 340+ characters
    normalized_id = normalize_point_cloud_id(long_id)
    
    assert isinstance(normalized_id, str)
    assert len(normalized_id) > 0


def test_normalize_point_cloud_id_deterministic():
    """Test that normalization is deterministic."""
    original_id = "test_point_cloud_id"
    
    normalized_1 = normalize_point_cloud_id(original_id)
    normalized_2 = normalize_point_cloud_id(original_id)
    
    assert normalized_1 == normalized_2


# ================================================================================
# point_cloud_to_numpy Tests - Valid Cases
# ================================================================================

def test_point_cloud_to_numpy_basic(point_cloud_3d):
    """Test basic point cloud to numpy conversion."""
    pc_numpy = point_cloud_to_numpy(point_cloud_3d)
    
    assert isinstance(pc_numpy, np.ndarray)
    assert pc_numpy.shape == (1000, 3)
    assert pc_numpy.dtype == np.float32  # Preserves original torch tensor dtype


def test_point_cloud_to_numpy_various_sizes():
    """Test conversion with various point cloud sizes."""
    sizes = [1, 10, 100, 1000]
    
    for size in sizes:
        pc = torch.randn(size, 3, dtype=torch.float32)
        pc_numpy = point_cloud_to_numpy(pc)
        
        assert isinstance(pc_numpy, np.ndarray)
        assert pc_numpy.shape == (size, 3)


def test_point_cloud_to_numpy_different_dtypes():
    """Test conversion with different tensor dtypes."""
    # Float32
    pc_f32 = torch.randn(100, 3, dtype=torch.float32)
    numpy_f32 = point_cloud_to_numpy(pc_f32)
    assert isinstance(numpy_f32, np.ndarray)
    assert numpy_f32.shape == (100, 3)
    
    # Float64
    pc_f64 = torch.randn(100, 3, dtype=torch.float64)
    numpy_f64 = point_cloud_to_numpy(pc_f64)
    assert isinstance(numpy_f64, np.ndarray)
    assert numpy_f64.shape == (100, 3)
    
    # Integer
    pc_int = torch.randint(-10, 10, (100, 3), dtype=torch.int32)
    numpy_int = point_cloud_to_numpy(pc_int)
    assert isinstance(numpy_int, np.ndarray)
    assert numpy_int.shape == (100, 3)


def test_point_cloud_to_numpy_single_point():
    """Test conversion with single point."""
    single_point = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
    pc_numpy = point_cloud_to_numpy(single_point)
    
    assert isinstance(pc_numpy, np.ndarray)
    assert pc_numpy.shape == (1, 3)
    assert np.allclose(pc_numpy, [[1.5, 2.5, 3.5]])


def test_point_cloud_to_numpy_values_preserved():
    """Test that conversion preserves values correctly."""
    # Known values
    pc_tensor = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=torch.float32)
    
    pc_numpy = point_cloud_to_numpy(pc_tensor)
    
    expected = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    assert isinstance(pc_numpy, np.ndarray)
    assert np.allclose(pc_numpy, expected)


def test_point_cloud_to_numpy_extreme_values():
    """Test conversion with extreme coordinate values."""
    # Very large values
    large_pc = torch.full((100, 3), 1e6, dtype=torch.float32)
    numpy_large = point_cloud_to_numpy(large_pc)
    assert isinstance(numpy_large, np.ndarray)
    assert numpy_large.shape == (100, 3)
    
    # Very small values
    small_pc = torch.full((100, 3), 1e-6, dtype=torch.float32)
    numpy_small = point_cloud_to_numpy(small_pc)
    assert isinstance(numpy_small, np.ndarray)
    assert numpy_small.shape == (100, 3)
    
    # Mixed values
    mixed_pc = torch.randn(100, 3, dtype=torch.float32) * 1000
    numpy_mixed = point_cloud_to_numpy(mixed_pc)
    assert isinstance(numpy_mixed, np.ndarray)
    assert numpy_mixed.shape == (100, 3)


# ================================================================================
# Integration and Performance Tests
# ================================================================================

def test_point_cloud_utilities_pipeline(point_cloud_3d, camera_state):
    """Test complete point cloud utilities pipeline."""
    # Generate ID (requires proper datapoint format)
    datapoint = {"meta_info": {"idx": 42}}
    pc_id = build_point_cloud_id(datapoint, "source")
    assert isinstance(pc_id, tuple)
    
    # Normalize ID
    normalized_id = normalize_point_cloud_id(pc_id)
    assert isinstance(normalized_id, str)
    
    # Get statistics
    stats = get_point_cloud_display_stats(point_cloud_3d)
    assert isinstance(stats, html.Ul)
    
    # Apply LOD (no max_points parameter)
    lod_points, lod_colors, lod_labels = apply_lod_to_point_cloud(
        points=point_cloud_3d,
        camera_state=camera_state,
        lod_type="continuous"
    )
    assert isinstance(lod_points, torch.Tensor)
    
    # Convert to numpy
    pc_numpy = point_cloud_to_numpy(lod_points)
    assert isinstance(pc_numpy, np.ndarray)
    
    # Verify consistency
    assert lod_points.shape[0] == pc_numpy.shape[0]
    assert lod_points.shape[1] == pc_numpy.shape[1] == 3


def test_point_cloud_utilities_determinism(point_cloud_3d, camera_state):
    """Test that point cloud utilities are deterministic."""
    # Multiple calls should produce same results
    datapoint = {"meta_info": {"idx": 42}}
    pc_id_1 = build_point_cloud_id(datapoint, "source")
    pc_id_2 = build_point_cloud_id(datapoint, "source")
    assert pc_id_1 == pc_id_2
    
    stats_1 = get_point_cloud_display_stats(point_cloud_3d)
    stats_2 = get_point_cloud_display_stats(point_cloud_3d)
    assert str(stats_1) == str(stats_2)  # Compare HTML strings since Ul objects are different instances
    
    normalized_1 = normalize_point_cloud_id("test_id")
    normalized_2 = normalize_point_cloud_id("test_id")
    assert normalized_1 == normalized_2
    
    numpy_1 = point_cloud_to_numpy(point_cloud_3d)
    numpy_2 = point_cloud_to_numpy(point_cloud_3d)
    assert np.array_equal(numpy_1, numpy_2)


def test_get_point_cloud_display_stats_with_4_channels():
    """Test that get_point_cloud_display_stats accepts 4+ channels (valid case)."""
    # 4 channels should be VALID - function accepts [N, 3+] 
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    stats = get_point_cloud_display_stats(pc_4ch)
    
    assert isinstance(stats, html.Ul)
    stats_text = str(stats)
    assert "Total Points: 100" in stats_text
    assert "Dimensions: 4" in stats_text  # Should show 4 dimensions


def test_get_point_cloud_display_stats_with_6_channels():
    """Test that get_point_cloud_display_stats accepts 6 channels (valid case)."""
    # 6 channels should be VALID - function accepts [N, 3+]
    pc_6ch = torch.randn(50, 6, dtype=torch.float32)
    stats = get_point_cloud_display_stats(pc_6ch)
    
    assert isinstance(stats, html.Ul)
    stats_text = str(stats)
    assert "Total Points: 50" in stats_text
    assert "Dimensions: 6" in stats_text  # Should show 6 dimensions


def test_performance_with_large_point_clouds():
    """Test utilities performance with large point clouds."""
    # Create large point cloud
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    # These should complete without error
    datapoint = {"meta_info": {"idx": 100}}
    pc_id = build_point_cloud_id(datapoint, "large_test")
    stats = get_point_cloud_display_stats(large_pc)
    lod_points, lod_colors, lod_labels = apply_lod_to_point_cloud(
        points=large_pc,
        camera_state=camera_state,
        lod_type="continuous"
    )
    pc_numpy = point_cloud_to_numpy(lod_points)
    
    # Basic checks
    assert isinstance(pc_id, tuple)
    assert isinstance(stats, html.Ul)
    assert isinstance(lod_points, torch.Tensor)
    assert isinstance(pc_numpy, np.ndarray)
    assert lod_points.shape[0] <= large_pc.shape[0]  # LOD should reduce or maintain size
    assert pc_numpy.shape == lod_points.shape
