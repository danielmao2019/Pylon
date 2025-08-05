"""Tests for point cloud display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

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
    
    assert isinstance(stats, dict)
    assert "num_points" in stats
    assert "x_range" in stats
    assert "y_range" in stats
    assert "z_range" in stats
    assert "centroid" in stats
    assert "bounding_box_volume" in stats
    
    # Verify basic properties
    assert stats["num_points"] == 1000


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
    
    assert isinstance(stats, dict)
    assert stats["num_points"] == 100
    
    # Check ranges
    assert "[0.000, 2.000]" in stats["x_range"]
    assert "[0.000, 3.000]" in stats["y_range"]
    assert "[0.000, 4.000]" in stats["z_range"]


def test_get_point_cloud_display_stats_single_point():
    """Test statistics with single point."""
    single_point = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(single_point)
    
    assert isinstance(stats, dict)
    assert stats["num_points"] == 1
    assert "[1.500, 1.500]" in stats["x_range"]
    assert "[2.500, 2.500]" in stats["y_range"]  
    assert "[3.500, 3.500]" in stats["z_range"]
    assert stats["bounding_box_volume"] == 0.0  # Single point has no volume


def test_get_point_cloud_display_stats_uniform_distribution():
    """Test statistics with uniformly distributed points."""
    # Create points in unit cube [0,1]^3
    points = torch.rand(1000, 3, dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, dict)
    assert stats["num_points"] == 1000
    
    # Ranges should be approximately [0, 1] for each dimension
    # (allowing some randomness tolerance)
    assert stats["bounding_box_volume"] > 0.0  # Should have positive volume


@pytest.mark.parametrize("n_points", [10, 100, 1000, 5000])
def test_get_point_cloud_display_stats_various_sizes(n_points):
    """Test statistics with various point cloud sizes."""
    points = torch.randn(n_points, 3, dtype=torch.float32)
    
    stats = get_point_cloud_display_stats(points)
    
    assert isinstance(stats, dict)
    assert stats["num_points"] == n_points


def test_get_point_cloud_display_stats_different_dtypes():
    """Test statistics with different tensor dtypes."""
    # Float32
    points_f32 = torch.randn(100, 3, dtype=torch.float32)
    stats_f32 = get_point_cloud_display_stats(points_f32)
    assert isinstance(stats_f32, dict)
    
    # Float64
    points_f64 = torch.randn(100, 3, dtype=torch.float64)
    stats_f64 = get_point_cloud_display_stats(points_f64)
    assert isinstance(stats_f64, dict)
    
    # Integer (unusual but should work)
    points_int = torch.randint(-10, 10, (100, 3), dtype=torch.int32)
    stats_int = get_point_cloud_display_stats(points_int)
    assert isinstance(stats_int, dict)


def test_get_point_cloud_display_stats_extreme_coordinates():
    """Test statistics with extreme coordinate values."""
    # Very large coordinates
    large_points = torch.full((100, 3), 1e6, dtype=torch.float32)
    stats_large = get_point_cloud_display_stats(large_points)
    assert isinstance(stats_large, dict)
    assert stats_large["bounding_box_volume"] == 0.0  # All points same location
    
    # Very small coordinates
    small_points = torch.full((100, 3), 1e-6, dtype=torch.float32)
    stats_small = get_point_cloud_display_stats(small_points)
    assert isinstance(stats_small, dict)
    
    # Mixed positive/negative
    mixed_points = torch.randn(100, 3, dtype=torch.float32) * 1000
    stats_mixed = get_point_cloud_display_stats(mixed_points)
    assert isinstance(stats_mixed, dict)


# ================================================================================
# build_point_cloud_id Tests - Valid Cases
# ================================================================================

def test_build_point_cloud_id_basic(point_cloud_3d):
    """Test basic point cloud ID generation."""
    pc_id = build_point_cloud_id(point_cloud_3d)
    
    assert isinstance(pc_id, str)
    assert len(pc_id) > 0


def test_build_point_cloud_id_deterministic(point_cloud_3d):
    """Test that point cloud ID generation is deterministic."""
    pc_id1 = build_point_cloud_id(point_cloud_3d)
    pc_id2 = build_point_cloud_id(point_cloud_3d)
    
    assert isinstance(pc_id1, str)
    assert isinstance(pc_id2, str)
    assert pc_id1 == pc_id2  # Should be identical for same input


def test_build_point_cloud_id_different_inputs():
    """Test that different point clouds generate different IDs."""
    pc1 = torch.randn(100, 3, dtype=torch.float32, generator=torch.Generator().manual_seed(42))
    pc2 = torch.randn(100, 3, dtype=torch.float32, generator=torch.Generator().manual_seed(123))
    
    id1 = build_point_cloud_id(pc1)
    id2 = build_point_cloud_id(pc2)
    
    assert isinstance(id1, str)
    assert isinstance(id2, str)
    assert id1 != id2  # Different inputs should generate different IDs


def test_build_point_cloud_id_various_sizes():
    """Test point cloud ID generation with various sizes."""
    sizes = [1, 10, 100, 1000]
    ids = []
    
    for size in sizes:
        pc = torch.randn(size, 3, dtype=torch.float32)
        pc_id = build_point_cloud_id(pc)
        
        assert isinstance(pc_id, str)
        assert len(pc_id) > 0
        ids.append(pc_id)
    
    # All IDs should be different
    assert len(set(ids)) == len(ids)


def test_build_point_cloud_id_single_point():
    """Test point cloud ID generation with single point."""
    single_point = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    pc_id = build_point_cloud_id(single_point)
    
    assert isinstance(pc_id, str)
    assert len(pc_id) > 0


def test_build_point_cloud_id_different_dtypes():
    """Test point cloud ID generation with different dtypes."""
    base_points = torch.randn(100, 3)
    
    # Float32
    points_f32 = base_points.to(torch.float32)
    id_f32 = build_point_cloud_id(points_f32)
    assert isinstance(id_f32, str)
    
    # Float64
    points_f64 = base_points.to(torch.float64)
    id_f64 = build_point_cloud_id(points_f64)
    assert isinstance(id_f64, str)
    
    # Different dtypes of same data might generate different IDs
    # (this is acceptable behavior)


# ================================================================================
# apply_lod_to_point_cloud Tests - Valid Cases
# ================================================================================

def test_apply_lod_to_point_cloud_basic(point_cloud_3d, camera_state):
    """Test basic LOD application."""
    lod_pc = apply_lod_to_point_cloud(point_cloud_3d, camera_state, max_points=500)
    
    assert isinstance(lod_pc, torch.Tensor)
    assert lod_pc.shape[1] == 3  # Should maintain 3D coordinates
    assert lod_pc.shape[0] <= 500  # Should not exceed max_points
    assert lod_pc.shape[0] <= point_cloud_3d.shape[0]  # Should not exceed original


def test_apply_lod_to_point_cloud_no_reduction_needed(camera_state):
    """Test LOD when no reduction is needed."""
    small_pc = torch.randn(50, 3, dtype=torch.float32)
    lod_pc = apply_lod_to_point_cloud(small_pc, camera_state, max_points=100)
    
    # Should return original point cloud (or very similar) since no reduction needed
    assert isinstance(lod_pc, torch.Tensor)
    assert lod_pc.shape[0] == 50  # Should keep all points
    assert lod_pc.shape[1] == 3


def test_apply_lod_to_point_cloud_max_points_one(point_cloud_3d, camera_state):
    """Test LOD with max_points=1."""
    lod_pc = apply_lod_to_point_cloud(point_cloud_3d, camera_state, max_points=1)
    
    assert isinstance(lod_pc, torch.Tensor)
    assert lod_pc.shape == (1, 3)  # Should return exactly one point


def test_apply_lod_to_point_cloud_various_max_points(point_cloud_3d, camera_state):
    """Test LOD with various max_points values."""
    max_points_values = [10, 50, 100, 500, 2000]
    
    for max_points in max_points_values:
        lod_pc = apply_lod_to_point_cloud(point_cloud_3d, camera_state, max_points=max_points)
        
        assert isinstance(lod_pc, torch.Tensor)
        assert lod_pc.shape[1] == 3
        assert lod_pc.shape[0] <= max_points
        assert lod_pc.shape[0] <= point_cloud_3d.shape[0]


def test_apply_lod_to_point_cloud_different_camera_positions():
    """Test LOD with different camera positions."""
    pc = torch.randn(1000, 3, dtype=torch.float32) * 10.0  # Spread out points
    
    # Camera at origin
    camera_origin = {"eye": {"x": 0, "y": 0, "z": 0}}
    lod_origin = apply_lod_to_point_cloud(pc, camera_origin, max_points=500)
    
    # Camera far away
    camera_far = {"eye": {"x": 100, "y": 100, "z": 100}}
    lod_far = apply_lod_to_point_cloud(pc, camera_far, max_points=500)
    
    # Both should return valid results
    assert isinstance(lod_origin, torch.Tensor)
    assert isinstance(lod_far, torch.Tensor)
    assert lod_origin.shape[1] == 3
    assert lod_far.shape[1] == 3


def test_apply_lod_to_point_cloud_single_point(camera_state):
    """Test LOD with single point."""
    single_point = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    lod_pc = apply_lod_to_point_cloud(single_point, camera_state, max_points=10)
    
    assert isinstance(lod_pc, torch.Tensor)
    assert torch.allclose(lod_pc, single_point)  # Should return the same point


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
    assert pc_numpy.dtype == np.float64  # Default numpy float type


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
    # Generate ID
    pc_id = build_point_cloud_id(point_cloud_3d)
    assert isinstance(pc_id, str)
    
    # Normalize ID
    normalized_id = normalize_point_cloud_id(pc_id)
    assert isinstance(normalized_id, str)
    
    # Get statistics
    stats = get_point_cloud_display_stats(point_cloud_3d)
    assert isinstance(stats, dict)
    
    # Apply LOD
    lod_pc = apply_lod_to_point_cloud(point_cloud_3d, camera_state, max_points=500)
    assert isinstance(lod_pc, torch.Tensor)
    
    # Convert to numpy
    pc_numpy = point_cloud_to_numpy(lod_pc)
    assert isinstance(pc_numpy, np.ndarray)
    
    # Verify consistency
    assert lod_pc.shape[0] == pc_numpy.shape[0]
    assert lod_pc.shape[1] == pc_numpy.shape[1] == 3


def test_point_cloud_utilities_determinism(point_cloud_3d, camera_state):
    """Test that point cloud utilities are deterministic."""
    # Multiple calls should produce same results
    pc_id_1 = build_point_cloud_id(point_cloud_3d)
    pc_id_2 = build_point_cloud_id(point_cloud_3d)
    assert pc_id_1 == pc_id_2
    
    stats_1 = get_point_cloud_display_stats(point_cloud_3d)
    stats_2 = get_point_cloud_display_stats(point_cloud_3d)
    assert stats_1 == stats_2
    
    normalized_1 = normalize_point_cloud_id("test_id")
    normalized_2 = normalize_point_cloud_id("test_id")
    assert normalized_1 == normalized_2
    
    numpy_1 = point_cloud_to_numpy(point_cloud_3d)
    numpy_2 = point_cloud_to_numpy(point_cloud_3d)
    assert np.array_equal(numpy_1, numpy_2)


def test_performance_with_large_point_clouds():
    """Test utilities performance with large point clouds."""
    # Create large point cloud
    large_pc = torch.randn(10000, 3, dtype=torch.float32)
    camera_state = {"eye": {"x": 1, "y": 1, "z": 1}}
    
    # These should complete without error
    pc_id = build_point_cloud_id(large_pc)
    stats = get_point_cloud_display_stats(large_pc)
    lod_pc = apply_lod_to_point_cloud(large_pc, camera_state, max_points=1000)
    pc_numpy = point_cloud_to_numpy(lod_pc)
    
    # Basic checks
    assert isinstance(pc_id, str)
    assert isinstance(stats, dict)
    assert isinstance(lod_pc, torch.Tensor)
    assert isinstance(pc_numpy, np.ndarray)
    assert lod_pc.shape[0] <= 1000  # LOD should reduce size
    assert pc_numpy.shape == lod_pc.shape