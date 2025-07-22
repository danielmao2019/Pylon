#!/usr/bin/env python3
"""
Test cases for OcclusionCrop transform.
"""

import pytest
import torch
import numpy as np
from data.transforms.vision_3d import OcclusionCrop


def test_occlusion_crop_initialization():
    """Test OcclusionCrop initialization with valid parameters."""
    # Test default initialization
    occlusion_crop = OcclusionCrop()
    assert occlusion_crop.ray_density_factor == 0.8
    
    # Test custom initialization
    occlusion_crop = OcclusionCrop(ray_density_factor=0.5)
    assert occlusion_crop.ray_density_factor == 0.5


def test_occlusion_crop_invalid_parameters():
    """Test OcclusionCrop initialization with invalid parameters."""
    # Test negative ray_density_factor
    with pytest.raises(AssertionError, match="ray_density_factor must be in \\[0.1, 1.0\\]"):
        OcclusionCrop(ray_density_factor=-1.0)
    
    # Test zero ray_density_factor
    with pytest.raises(AssertionError, match="ray_density_factor must be in \\[0.1, 1.0\\]"):
        OcclusionCrop(ray_density_factor=0.0)
    
    # Test ray_density_factor > 1.0
    with pytest.raises(AssertionError, match="ray_density_factor must be in \\[0.1, 1.0\\]"):
        OcclusionCrop(ray_density_factor=1.5)
    
    # Test non-numeric ray_density_factor
    with pytest.raises(AssertionError, match="ray_density_factor must be numeric"):
        OcclusionCrop(ray_density_factor="invalid")


def test_occlusion_crop_basic_functionality():
    """Test basic occlusion cropping functionality."""
    # Create a simple scene with occluding geometry
    # Points in a line: some close (occluding), some far (occluded)
    points = torch.tensor([
        [1.0, 0.0, 0.0],   # Close point (should be visible)
        [2.0, 0.0, 0.0],   # Farther point in same direction (should be occluded)
        [3.0, 0.0, 0.0],   # Even farther (should be occluded)
        [1.0, 1.0, 0.0],   # Close point in different direction (should be visible)
        [1.0, 0.0, 1.0],   # Close point in different direction (should be visible)
        [0.5, 0.0, 0.0],   # Very close point (should be visible)
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Sensor position at origin
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should keep some points (at least the closest ones)
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)
    assert result['feat'].shape[0] == len(result['pos'])


def test_occlusion_crop_no_occlusion():
    """Test occlusion cropping with well-separated points (no occlusion)."""
    # Create points in different directions from sensor (no occlusion)
    points = torch.tensor([
        [5.0, 0.0, 0.0],   # +X direction
        [0.0, 5.0, 0.0],   # +Y direction
        [0.0, 0.0, 5.0],   # +Z direction
        [-5.0, 0.0, 0.0],  # -X direction
        [0.0, -5.0, 0.0],  # -Y direction
        [0.0, 0.0, -5.0],  # -Z direction
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should keep all points (no occlusion)
    assert len(result['pos']) == len(points)


def test_occlusion_crop_clear_occlusion():
    """Test occlusion cropping with clear occlusion pattern."""
    # Create a clear occlusion scenario: wall with points behind it
    points = []
    
    # Create a "wall" of points at X=2
    for y in np.linspace(-1, 1, 5):
        for z in np.linspace(-1, 1, 5):
            points.append([2.0, y, z])
    
    # Add points behind the wall at X=4 (should be occluded)
    for y in np.linspace(-0.5, 0.5, 3):
        for z in np.linspace(-0.5, 0.5, 3):
            points.append([4.0, y, z])
    
    points = torch.tensor(points, dtype=torch.float32)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should keep mostly the wall points, fewer behind-wall points
    assert len(result['pos']) > 0
    assert len(result['pos']) < len(points)  # Some occlusion should occur
    
    # Most kept points should be from the wall (closer to sensor)
    kept_distances = torch.norm(result['pos'] - sensor_pos.unsqueeze(0), dim=1)
    assert torch.mean(kept_distances) < 3.0  # Average distance should be closer to wall than behind


def test_occlusion_crop_different_sensor_positions():
    """Test occlusion cropping with different sensor positions."""
    # Create points in a line
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test with sensor at origin
    sensor_pos1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    occlusion_crop = OcclusionCrop()
    result1 = occlusion_crop._call_single(pc, sensor_pos1)
    
    # Test with sensor at different position
    sensor_pos2 = torch.tensor([0.0, 5.0, 0.0], dtype=torch.float32)  # Move sensor to Y=5
    result2 = occlusion_crop._call_single(pc, sensor_pos2)
    
    # Results should be different due to different viewing angles
    # From Y=5, all points should be visible (no occlusion from this angle)
    assert len(result2['pos']) >= len(result1['pos'])


def test_occlusion_crop_density_analysis():
    """Test that density analysis works correctly."""
    # Create clusters of points with different densities
    dense_cluster = torch.randn(50, 3) * 0.1 + torch.tensor([1.0, 0.0, 0.0])  # Dense cluster
    sparse_cluster = torch.randn(20, 3) * 0.5 + torch.tensor([5.0, 0.0, 0.0])  # Sparse cluster
    
    points = torch.cat([dense_cluster, sparse_cluster], dim=0)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should handle mixed density scenarios
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)


def test_occlusion_crop_edge_cases():
    """Test edge cases for occlusion cropping."""
    occlusion_crop = OcclusionCrop()
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Test empty point cloud
    empty_pc = {'pos': torch.empty(0, 3, dtype=torch.float32), 'feat': torch.empty(0, 1)}
    result = occlusion_crop._call_single(empty_pc, sensor_pos)
    assert len(result['pos']) == 0
    assert len(result['feat']) == 0
    
    # Test single point
    single_pc = {'pos': torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32), 
                 'feat': torch.ones(1, 1)}
    result = occlusion_crop._call_single(single_pc, sensor_pos)
    assert len(result['pos']) == 1  # Single point should always be visible
    
    # Test two points (no occlusion possible with only 2 points in different directions)
    two_pc = {'pos': torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
              'feat': torch.ones(2, 1)}
    result = occlusion_crop._call_single(two_pc, sensor_pos)
    assert len(result['pos']) == 2  # Both should be visible


def test_occlusion_crop_small_point_clouds():
    """Test occlusion cropping with small point clouds (< 10 points)."""
    # Small point cloud (< k_neighbors default of 10)
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [3.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should handle small point clouds gracefully
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)


def test_occlusion_crop_device_handling():
    """Test that OcclusionCrop handles different devices correctly."""
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    occlusion_crop = OcclusionCrop()
    
    # Test with CPU tensors
    sensor_pos_cpu = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    result = occlusion_crop._call_single(pc, sensor_pos_cpu)
    assert result['pos'].device == points.device
    assert len(result['pos']) > 0
    
    # Test with mismatched devices
    if torch.cuda.is_available():
        sensor_pos_cuda = sensor_pos_cpu.cuda()
        result = occlusion_crop._call_single(pc, sensor_pos_cuda)
        assert result['pos'].device == points.device  # Should match input device
        assert len(result['pos']) > 0


def test_occlusion_crop_multiple_features():
    """Test occlusion cropping with multiple feature channels."""
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    # Test with multiple feature types
    pc = {
        'pos': points,
        'feat': torch.randn(4, 5),  # 5-dimensional features
        'colors': torch.randn(4, 3),  # RGB colors
        'normals': torch.randn(4, 3)  # Surface normals
    }
    
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should keep some points
    num_kept = len(result['pos'])
    assert num_kept > 0
    assert result['feat'].shape == (num_kept, 5)
    assert result['colors'].shape == (num_kept, 3)
    assert result['normals'].shape == (num_kept, 3)
    
    # Features should correspond to kept points (check shapes match)
    assert result['feat'].shape[0] == result['pos'].shape[0]
    assert result['colors'].shape[0] == result['pos'].shape[0]
    assert result['normals'].shape[0] == result['pos'].shape[0]


def test_occlusion_crop_deterministic():
    """Test that occlusion cropping is deterministic."""
    # Use fixed seed for reproducible random points
    torch.manual_seed(42)
    points = torch.randn(50, 3) * 5  # Random points
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    
    # Run multiple times
    result1 = occlusion_crop._call_single(pc, sensor_pos)
    result2 = occlusion_crop._call_single(pc, sensor_pos)
    
    # Results should be identical
    assert len(result1['pos']) == len(result2['pos'])
    assert torch.allclose(result1['pos'], result2['pos'])


def test_occlusion_crop_ray_density_factor():
    """Test that ray_density_factor affects voxel size appropriately."""
    # Create a controlled scene
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    # Test with smaller ray_density_factor
    occlusion_crop_fine = OcclusionCrop(ray_density_factor=0.1)
    result_fine = occlusion_crop_fine._call_single(pc, sensor_pos)
    
    # Test with larger ray_density_factor
    occlusion_crop_coarse = OcclusionCrop(ray_density_factor=1.0)
    result_coarse = occlusion_crop_coarse._call_single(pc, sensor_pos)
    
    # Both should return some points
    assert len(result_fine['pos']) > 0
    assert len(result_coarse['pos']) > 0
    
    # Results might be different due to different ray density factors
    # (but we can't guarantee which will have more points without knowing the exact scene)


def test_occlusion_crop_large_point_cloud():
    """Test occlusion cropping performance with larger point clouds."""
    # Create a larger point cloud for performance testing
    torch.manual_seed(42)
    points = torch.randn(500, 3) * 10  # 500 random points
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop()
    
    # Should complete without errors
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)
    assert result['pos'].shape[1] == 3  # 3D coordinates
    assert result['feat'].shape == (len(result['pos']), 1)


@pytest.mark.parametrize("ray_density_factor", [0.1, 0.3, 0.5, 0.8, 1.0])
def test_occlusion_crop_parametrized_density_factor(ray_density_factor):
    """Test occlusion cropping with different ray_density_factor values."""
    # Create consistent test scene
    points = torch.tensor([
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    occlusion_crop = OcclusionCrop(ray_density_factor=ray_density_factor)
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should work for all density factors
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)


def test_occlusion_crop_input_validation():
    """Test input validation for _call_single method."""
    occlusion_crop = OcclusionCrop()
    points = torch.randn(10, 3)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test invalid sensor_pos type
    with pytest.raises(AssertionError, match="sensor_pos must be torch.Tensor"):
        occlusion_crop._call_single(pc, [0.0, 0.0, 0.0])
    
    # Test invalid sensor_pos shape
    with pytest.raises(AssertionError, match="sensor_pos must be \\[3\\]"):
        wrong_shape = torch.eye(4, dtype=torch.float32)
        occlusion_crop._call_single(pc, wrong_shape)
    
    # Test invalid point cloud (missing 'pos')
    with pytest.raises(AssertionError):
        invalid_pc = {'feat': torch.ones(10, 1)}
        sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        occlusion_crop._call_single(invalid_pc, sensor_pos)


def test_occlusion_crop_scipy_availability():
    """Test that OcclusionCrop properly handles scipy availability."""
    # This test ensures the import works and the method can be called
    # If scipy is not available, the class construction should fail during import
    points = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    # If we get here, scipy is available
    occlusion_crop = OcclusionCrop()
    result = occlusion_crop._call_single(pc, sensor_pos)
    
    # Should work correctly
    assert len(result['pos']) > 0
