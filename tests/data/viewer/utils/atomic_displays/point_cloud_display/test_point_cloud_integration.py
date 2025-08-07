"""Tests for point cloud integration and performance scenarios.

Focuses on integration testing between multiple point cloud utilities and performance tests.
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
# Integration Tests
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
    assert isinstance(stats, dict)
    
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
    assert stats_1 == stats_2  # Dictionaries should be equal
    
    normalized_1 = normalize_point_cloud_id("test_id")
    normalized_2 = normalize_point_cloud_id("test_id")
    assert normalized_1 == normalized_2
    
    numpy_1 = point_cloud_to_numpy(point_cloud_3d)
    numpy_2 = point_cloud_to_numpy(point_cloud_3d)
    assert np.array_equal(numpy_1, numpy_2)


# ================================================================================
# Performance Tests
# ================================================================================

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
    assert isinstance(stats, dict)
    assert isinstance(lod_points, torch.Tensor)
    assert isinstance(pc_numpy, np.ndarray)
    assert lod_points.shape[0] <= large_pc.shape[0]  # LOD should reduce or maintain size
    assert pc_numpy.shape == lod_points.shape