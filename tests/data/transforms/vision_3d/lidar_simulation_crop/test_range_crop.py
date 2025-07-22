#!/usr/bin/env python3
"""
Test cases for RangeCrop transform.
"""

import pytest
import torch
import numpy as np
from data.transforms.vision_3d import RangeCrop


def test_range_crop_initialization():
    """Test RangeCrop initialization with valid parameters."""
    # Test default initialization
    range_crop = RangeCrop()
    assert range_crop.max_range == 100.0
    
    # Test custom initialization
    range_crop = RangeCrop(max_range=50.5)
    assert range_crop.max_range == 50.5


def test_range_crop_invalid_parameters():
    """Test RangeCrop initialization with invalid parameters."""
    # Test negative max_range
    with pytest.raises(AssertionError, match="max_range must be positive"):
        RangeCrop(max_range=-1.0)
    
    # Test zero max_range
    with pytest.raises(AssertionError, match="max_range must be positive"):
        RangeCrop(max_range=0.0)
    
    # Test non-numeric max_range
    with pytest.raises(AssertionError, match="max_range must be numeric"):
        RangeCrop(max_range="invalid")


def test_range_crop_basic_functionality():
    """Test basic range cropping functionality."""
    # Create test point cloud - points in a line along X axis
    points = torch.tensor([
        [1.0, 0.0, 0.0],   # Within range
        [3.0, 0.0, 0.0],   # Within range
        [5.0, 0.0, 0.0],   # At boundary
        [7.0, 0.0, 0.0],   # Beyond range
        [10.0, 0.0, 0.0],  # Far beyond range
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Test with max_range = 5.0
    range_crop = RangeCrop(max_range=5.0)
    result = range_crop._call_single(pc, sensor_pos)
    
    # Should keep first 3 points (distances: 1, 3, 5)
    assert len(result['pos']) == 3
    assert torch.allclose(result['pos'], points[:3])
    assert result['feat'].shape == (3, 1)


def test_range_crop_different_sensor_positions():
    """Test range cropping with different sensor positions."""
    # Create test points
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    range_crop = RangeCrop(max_range=3.0)
    
    # Test with sensor at origin
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    result = range_crop._call_single(pc, sensor_pos)
    assert len(result['pos']) == 2  # Points at 0, 2
    
    # Test with sensor at [2, 0, 0]
    sensor_pos = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
    result = range_crop._call_single(pc, sensor_pos)
    assert len(result['pos']) == 3  # Points at 0, 2, 4 (distances: 2, 0, 2)


def test_range_crop_3d_distances():
    """Test range cropping with 3D point distributions."""
    # Create points at known 3D distances
    points = torch.tensor([
        [1.0, 0.0, 0.0],     # Distance: 1.0
        [0.0, 2.0, 0.0],     # Distance: 2.0
        [0.0, 0.0, 3.0],     # Distance: 3.0
        [2.0, 2.0, 0.0],     # Distance: 2.83
        [1.0, 1.0, 1.0],     # Distance: 1.73
        [3.0, 4.0, 0.0],     # Distance: 5.0
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    range_crop = RangeCrop(max_range=3.0)
    result = range_crop._call_single(pc, sensor_pos)
    
    # Should keep points with distances <= 3.0: [1.0, 2.0, 3.0, 2.83, 1.73]
    assert len(result['pos']) == 5
    
    # Verify the far point (3,4,0) is excluded
    excluded_point = torch.tensor([3.0, 4.0, 0.0])
    assert not torch.any(torch.all(result['pos'] == excluded_point, dim=1))


def test_range_crop_edge_cases():
    """Test edge cases for range cropping."""
    range_crop = RangeCrop(max_range=5.0)
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Test empty point cloud
    empty_pc = {'pos': torch.empty(0, 3, dtype=torch.float32), 'feat': torch.empty(0, 1)}
    result = range_crop._call_single(empty_pc, sensor_pos)
    assert len(result['pos']) == 0
    assert len(result['feat']) == 0
    
    # Test single point within range
    single_pc = {'pos': torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32), 
                 'feat': torch.ones(1, 1)}
    result = range_crop._call_single(single_pc, sensor_pos)
    assert len(result['pos']) == 1
    
    # Test single point beyond range
    far_pc = {'pos': torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
              'feat': torch.ones(1, 1)}
    result = range_crop._call_single(far_pc, sensor_pos)
    assert len(result['pos']) == 0


def test_range_crop_device_handling():
    """Test that RangeCrop handles different devices correctly."""
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    range_crop = RangeCrop(max_range=5.0)
    
    # Test with CPU tensors
    sensor_pos_cpu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    result = range_crop._call_single(pc, sensor_pos_cpu)
    assert result['pos'].device == points.device
    assert len(result['pos']) == 1
    
    # Test with mismatched devices (sensor_pos different device)
    if torch.cuda.is_available():
        sensor_pos_cuda = sensor_pos_cpu.cuda()
        result = range_crop._call_single(pc, sensor_pos_cuda)
        assert result['pos'].device == points.device  # Should match input device
        assert len(result['pos']) == 1


def test_range_crop_multiple_features():
    """Test range cropping with multiple feature channels."""
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0], 
        [7.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    # Test with multiple feature types
    pc = {
        'pos': points,
        'feat': torch.randn(3, 5),  # 5-dimensional features
        'colors': torch.randn(3, 3),  # RGB colors
        'normals': torch.randn(3, 3)  # Surface normals
    }
    
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    range_crop = RangeCrop(max_range=5.0)
    result = range_crop._call_single(pc, sensor_pos)
    
    # Should keep first 2 points
    assert len(result['pos']) == 2
    assert result['feat'].shape == (2, 5)
    assert result['colors'].shape == (2, 3)
    assert result['normals'].shape == (2, 3)
    
    # Verify features correspond to kept points
    assert torch.allclose(result['feat'], pc['feat'][:2])
    assert torch.allclose(result['colors'], pc['colors'][:2])
    assert torch.allclose(result['normals'], pc['normals'][:2])


def test_range_crop_deterministic():
    """Test that range cropping is deterministic."""
    points = torch.randn(100, 3) * 10  # Random points
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    range_crop = RangeCrop(max_range=5.0)
    
    # Run multiple times
    result1 = range_crop._call_single(pc, sensor_pos)
    result2 = range_crop._call_single(pc, sensor_pos)
    
    # Results should be identical
    assert len(result1['pos']) == len(result2['pos'])
    assert torch.allclose(result1['pos'], result2['pos'])


def test_range_crop_precision():
    """Test range cropping precision at boundary conditions."""
    # Create points exactly at boundary
    boundary_distance = 5.0
    points = torch.tensor([
        [boundary_distance - 1e-6, 0.0, 0.0],  # Just inside
        [boundary_distance, 0.0, 0.0],          # Exactly at boundary  
        [boundary_distance + 1e-6, 0.0, 0.0],  # Just outside
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    range_crop = RangeCrop(max_range=boundary_distance)
    result = range_crop._call_single(pc, sensor_pos)
    
    # Should keep first 2 points (inside and at boundary)
    assert len(result['pos']) == 2
    assert torch.allclose(result['pos'], points[:2])


@pytest.mark.parametrize("max_range", [0.1, 1.0, 10.0, 100.0])
def test_range_crop_parametrized_ranges(max_range):
    """Test range cropping with different max_range values."""
    # Create points at various distances
    points = torch.tensor([
        [0.05, 0.0, 0.0],  # Very close
        [0.5, 0.0, 0.0],   # Close
        [5.0, 0.0, 0.0],   # Medium
        [50.0, 0.0, 0.0],  # Far
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    range_crop = RangeCrop(max_range=max_range)
    result = range_crop._call_single(pc, sensor_pos)
    
    # Count expected points
    distances = torch.norm(points - sensor_pos, dim=1)
    expected_count = torch.sum(distances <= max_range).item()
    
    assert len(result['pos']) == expected_count


def test_range_crop_input_validation():
    """Test input validation for _call_single method."""
    range_crop = RangeCrop(max_range=5.0)
    points = torch.randn(10, 3)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test invalid sensor_pos type
    with pytest.raises(AssertionError, match="sensor_pos must be torch.Tensor"):
        range_crop._call_single(pc, [0.0, 0.0, 0.0])
    
    # Test invalid sensor_pos shape
    with pytest.raises(AssertionError, match="sensor_pos must be \\[3\\]"):
        sensor_pos_wrong_shape = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        range_crop._call_single(pc, sensor_pos_wrong_shape)
    
    # Test invalid point cloud (missing 'pos')
    with pytest.raises(AssertionError):
        invalid_pc = {'feat': torch.ones(10, 1)}
        sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        range_crop._call_single(invalid_pc, sensor_pos)