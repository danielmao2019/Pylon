"""Tests for point cloud conversion functionality.

Focuses specifically on the point_cloud_to_numpy function.
CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np

from data.viewer.utils.atomic_displays.point_cloud_display import (
    point_cloud_to_numpy
)


# ================================================================================
# point_cloud_to_numpy Tests
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