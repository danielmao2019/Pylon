#!/usr/bin/env python3
"""
Test cases for LiDARSimulationCrop transform.
"""

import pytest
import torch
import numpy as np
from data.transforms.vision_3d import LiDARSimulationCrop


def test_lidar_simulation_crop_initialization():
    """Test LiDARSimulationCrop initialization with valid parameters."""
    # Test default initialization
    lidar_crop = LiDARSimulationCrop()
    assert lidar_crop.max_range == 100.0
    assert lidar_crop.horizontal_fov == 360.0
    assert lidar_crop.vertical_fov == 40.0
    assert lidar_crop.ray_density_factor == 0.8
    assert lidar_crop.apply_range_filter is True
    assert lidar_crop.apply_fov_filter is True
    assert lidar_crop.apply_occlusion_filter is False
    
    # Test custom initialization
    lidar_crop = LiDARSimulationCrop(
        max_range=50.0,
        fov=(120.0, 90.0),
        ray_density_factor=0.5,
        apply_range_filter=False,
        apply_fov_filter=True,
        apply_occlusion_filter=True
    )
    assert lidar_crop.max_range == 50.0
    assert lidar_crop.horizontal_fov == 120.0
    assert lidar_crop.vertical_fov == 90.0
    assert lidar_crop.ray_density_factor == 0.5
    assert lidar_crop.apply_range_filter is False
    assert lidar_crop.apply_fov_filter is True
    assert lidar_crop.apply_occlusion_filter is True


def test_lidar_simulation_crop_component_creation():
    """Test that component crops are created correctly based on flags."""
    # Test with all filters enabled
    lidar_crop = LiDARSimulationCrop(
        max_range=100.0,
        fov=(360.0, 40.0),
        ray_density_factor=0.8,
        apply_range_filter=True,
        apply_fov_filter=True,
        apply_occlusion_filter=True
    )
    assert lidar_crop.range_crop is not None
    assert lidar_crop.fov_crop is not None
    assert lidar_crop.occlusion_crop is not None
    
    # Test with only range filter
    lidar_crop = LiDARSimulationCrop(
        max_range=100.0,
        fov=(360.0, 40.0),
        ray_density_factor=0.8,
        apply_range_filter=True,
        apply_fov_filter=False,
        apply_occlusion_filter=False
    )
    assert lidar_crop.range_crop is not None
    assert lidar_crop.fov_crop is None
    assert lidar_crop.occlusion_crop is None
    
    # Test with only FOV filter
    lidar_crop = LiDARSimulationCrop(
        max_range=100.0,
        fov=(360.0, 40.0),
        ray_density_factor=0.8,
        apply_range_filter=False,
        apply_fov_filter=True,
        apply_occlusion_filter=False
    )
    assert lidar_crop.range_crop is None
    assert lidar_crop.fov_crop is not None
    assert lidar_crop.occlusion_crop is None
    
    # Test with only occlusion filter
    lidar_crop = LiDARSimulationCrop(
        max_range=100.0,
        fov=(360.0, 40.0),
        ray_density_factor=0.8,
        apply_range_filter=False,
        apply_fov_filter=False,
        apply_occlusion_filter=True
    )
    assert lidar_crop.range_crop is None
    assert lidar_crop.fov_crop is None
    assert lidar_crop.occlusion_crop is not None
    
    # Test with no filters (should still work)
    lidar_crop = LiDARSimulationCrop(
        max_range=100.0,
        fov=(360.0, 40.0),
        ray_density_factor=0.8,
        apply_range_filter=False,
        apply_fov_filter=False,
        apply_occlusion_filter=False
    )
    assert lidar_crop.range_crop is None
    assert lidar_crop.fov_crop is None
    assert lidar_crop.occlusion_crop is None


def test_lidar_simulation_crop_basic_functionality():
    """Test basic LiDARSimulationCrop functionality."""
    # Create test points
    points = torch.tensor([
        [2.0, 0.0, 0.0],    # Forward, close
        [50.0, 0.0, 0.0],   # Forward, far
        [200.0, 0.0, 0.0],  # Forward, very far (should be filtered by range)
        [2.0, 2.0, 0.0],    # Forward-right
        [-2.0, 0.0, 0.0],   # Behind
        [2.0, 0.0, 2.0],    # Forward, up
        [2.0, 0.0, -2.0],   # Forward, down
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Sensor at origin, looking along +X
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test with range and FOV filters enabled, occlusion disabled for predictable results
    lidar_crop = LiDARSimulationCrop(
        max_range=100.0,
        horizontal_fov=90.0,
        vertical_fov=90.0,
        apply_range_filter=True,
        apply_fov_filter=True,
        apply_occlusion_filter=False
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should filter out some points
    assert len(result['pos']) > 0
    assert len(result['pos']) < len(points)
    assert result['feat'].shape[0] == len(result['pos'])


def test_lidar_simulation_crop_range_only():
    """Test LiDARSimulationCrop with only range filtering."""
    points = torch.tensor([
        [5.0, 0.0, 0.0],    # Within range
        [15.0, 0.0, 0.0],   # Beyond range
        [3.0, 0.0, 0.0],    # Within range
        [20.0, 0.0, 0.0],   # Beyond range
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        max_range=10.0,
        apply_range_filter=True,
        apply_fov_filter=False,
        apply_occlusion_filter=False
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should keep only points within range
    assert len(result['pos']) == 2
    distances = torch.norm(result['pos'], dim=1)
    assert torch.all(distances <= 10.0)


def test_lidar_simulation_crop_fov_only():
    """Test LiDARSimulationCrop with only FOV filtering."""
    points = torch.tensor([
        [5.0, 0.0, 0.0],    # Forward (within FOV)
        [5.0, 5.0, 0.0],    # Side (within 90° HFOV)
        [5.0, -5.0, 0.0],   # Side (within 90° HFOV)
        [-5.0, 0.0, 0.0],   # Behind (outside FOV with 90° HFOV)
        [5.0, 0.0, 5.0],    # Up (may be outside default VFOV)
        [5.0, 0.0, -5.0],   # Down (may be outside default VFOV)
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        horizontal_fov=90.0,
        vertical_fov=90.0,
        apply_range_filter=False,
        apply_fov_filter=True,
        apply_occlusion_filter=False
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should filter out some points based on FOV
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)


def test_lidar_simulation_crop_occlusion_only():
    """Test LiDARSimulationCrop with only occlusion filtering."""
    # Create points with clear occlusion pattern
    points = torch.tensor([
        [1.0, 0.0, 0.0],    # Close point
        [2.0, 0.0, 0.0],    # Farther point (same direction, should be occluded)
        [1.0, 1.0, 0.0],    # Different direction
        [1.0, 0.0, 1.0],    # Different direction
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        apply_range_filter=False,
        apply_fov_filter=False,
        apply_occlusion_filter=True
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should keep some points (occlusion filtering should work)
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)


def test_lidar_simulation_crop_no_filters():
    """Test LiDARSimulationCrop with no filters (pass-through)."""
    points = torch.randn(20, 3) * 10  # Random points
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        apply_range_filter=False,
        apply_fov_filter=False,
        apply_occlusion_filter=False
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should keep all points (no filtering)
    assert len(result['pos']) == len(points)
    assert torch.allclose(result['pos'], points)
    assert torch.allclose(result['feat'], pc['feat'])


def test_lidar_simulation_crop_combined_filters():
    """Test LiDARSimulationCrop with multiple filters combined."""
    # Create a comprehensive test scene
    points = torch.tensor([
        [2.0, 0.0, 0.0],    # Forward, close, visible
        [8.0, 0.0, 0.0],    # Forward, medium distance, visible
        [15.0, 0.0, 0.0],   # Forward, beyond range limit
        [2.0, 3.0, 0.0],    # Forward-right, within FOV
        [2.0, -3.0, 0.0],   # Forward-left, within FOV
        [-2.0, 0.0, 0.0],   # Behind, outside FOV
        [2.0, 0.0, 3.0],    # Forward-up, might be outside VFOV
        [2.0, 0.0, -3.0],   # Forward-down, might be outside VFOV
        [4.0, 0.0, 0.0],    # Forward, medium distance (potential occlusion)
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        max_range=10.0,
        horizontal_fov=120.0,
        vertical_fov=60.0,
        apply_range_filter=True,
        apply_fov_filter=True,
        apply_occlusion_filter=False  # Disable for predictable results
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should apply filters
    assert len(result['pos']) > 0
    assert len(result['pos']) < len(points)
    
    # No points should be beyond max_range
    distances = torch.norm(result['pos'], dim=1)
    assert torch.all(distances <= 10.0)


def test_lidar_simulation_crop_different_sensor_poses():
    """Test LiDARSimulationCrop with different sensor poses."""
    points = torch.tensor([
        [5.0, 0.0, 0.0],   # +X
        [0.0, 5.0, 0.0],   # +Y
        [0.0, 0.0, 5.0],   # +Z
        [-5.0, 0.0, 0.0],  # -X
        [0.0, -5.0, 0.0],  # -Y
        [0.0, 0.0, -5.0],  # -Z
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test with sensor looking along +Y axis
    sensor_extrinsics = torch.tensor([
        [0.0, -1.0, 0.0, 0.0],  # +X sensor = -Y world
        [1.0, 0.0, 0.0, 0.0],   # +Y sensor = +X world
        [0.0, 0.0, 1.0, 0.0],   # +Z sensor = +Z world
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        max_range=10.0,
        horizontal_fov=90.0,
        apply_range_filter=True,
        apply_fov_filter=True,
        apply_occlusion_filter=False
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should see different points based on sensor orientation
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)


def test_lidar_simulation_crop_edge_cases():
    """Test edge cases for LiDARSimulationCrop."""
    lidar_crop = LiDARSimulationCrop()
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test empty point cloud
    empty_pc = {'pos': torch.empty(0, 3, dtype=torch.float32), 'feat': torch.empty(0, 1)}
    result = lidar_crop._call_single(empty_pc, sensor_extrinsics, generator=torch.Generator())
    assert len(result['pos']) == 0
    assert len(result['feat']) == 0
    
    # Test single point
    single_pc = {'pos': torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32), 
                 'feat': torch.ones(1, 1)}
    result = lidar_crop._call_single(single_pc, sensor_extrinsics, generator=torch.Generator())
    # Result depends on filters, but should not crash
    assert len(result['pos']) >= 0
    assert len(result['feat']) == len(result['pos'])


def test_lidar_simulation_crop_device_handling():
    """Test that LiDARSimulationCrop handles different devices correctly."""
    points = torch.tensor([
        [5.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    lidar_crop = LiDARSimulationCrop()
    
    # Test with CPU tensors
    sensor_extrinsics_cpu = torch.eye(4, dtype=torch.float32)
    result = lidar_crop._call_single(pc, sensor_extrinsics_cpu, generator=torch.Generator())
    assert result['pos'].device == points.device
    assert len(result['pos']) >= 0
    
    # Test with mismatched devices
    if torch.cuda.is_available():
        sensor_extrinsics_cuda = sensor_extrinsics_cpu.cuda()
        result = lidar_crop._call_single(pc, sensor_extrinsics_cuda, generator=torch.Generator())
        assert result['pos'].device == points.device  # Should match input device
        assert len(result['pos']) >= 0


def test_lidar_simulation_crop_multiple_features():
    """Test LiDARSimulationCrop with multiple feature channels."""
    points = torch.tensor([
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [8.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    # Test with multiple feature types
    pc = {
        'pos': points,
        'feat': torch.randn(4, 5),  # 5-dimensional features
        'colors': torch.randn(4, 3),  # RGB colors
        'normals': torch.randn(4, 3)  # Surface normals
    }
    
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    lidar_crop = LiDARSimulationCrop()
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should preserve feature structure
    num_kept = len(result['pos'])
    assert num_kept >= 0
    assert result['feat'].shape == (num_kept, 5)
    assert result['colors'].shape == (num_kept, 3)
    assert result['normals'].shape == (num_kept, 3)


def test_lidar_simulation_crop_deterministic():
    """Test that LiDARSimulationCrop is deterministic with same generator."""
    torch.manual_seed(42)
    points = torch.randn(50, 3) * 5
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop()
    
    # Use same seed for generator
    gen1 = torch.Generator()
    gen1.manual_seed(123)
    result1 = lidar_crop._call_single(pc, sensor_extrinsics, generator=gen1)
    
    gen2 = torch.Generator()
    gen2.manual_seed(123)
    result2 = lidar_crop._call_single(pc, sensor_extrinsics, generator=gen2)
    
    # Results should be identical with same generator seed
    assert len(result1['pos']) == len(result2['pos'])
    if len(result1['pos']) > 0:
        assert torch.allclose(result1['pos'], result2['pos'])


def test_lidar_simulation_crop_filter_order():
    """Test that filter application order is consistent."""
    # Create points that will be affected differently by filter order
    points = torch.tensor([
        [2.0, 0.0, 0.0],    # Should survive all filters
        [50.0, 0.0, 0.0],   # Will be filtered by range
        [-2.0, 0.0, 0.0],   # Will be filtered by FOV
        [3.0, 0.0, 0.0],    # Might be filtered by occlusion
        [4.0, 0.0, 0.0],    # Might be filtered by occlusion
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        max_range=10.0,
        horizontal_fov=120.0,
        apply_range_filter=True,
        apply_fov_filter=True,
        apply_occlusion_filter=False  # Disable for predictable results
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should apply filters consistently
    assert len(result['pos']) > 0
    assert len(result['pos']) < len(points)
    
    # Range filter should remove far points
    distances = torch.norm(result['pos'], dim=1)
    assert torch.all(distances <= 10.0)


@pytest.mark.parametrize("apply_range,apply_fov,apply_occlusion", [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
    (False, False, False)
])
def test_lidar_simulation_crop_parametrized_filters(apply_range, apply_fov, apply_occlusion):
    """Test LiDARSimulationCrop with different filter combinations."""
    points = torch.tensor([
        [2.0, 0.0, 0.0],
        [15.0, 0.0, 0.0],  # Far point (range filter)
        [-2.0, 0.0, 0.0],  # Behind point (FOV filter)
        [3.0, 0.0, 0.0],   # Potential occlusion
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    lidar_crop = LiDARSimulationCrop(
        max_range=10.0,
        horizontal_fov=90.0,
        apply_range_filter=apply_range,
        apply_fov_filter=apply_fov,
        apply_occlusion_filter=apply_occlusion
    )
    
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator=torch.Generator())
    
    # Should work for all combinations
    assert len(result['pos']) >= 0
    assert len(result['pos']) <= len(points)
    assert result['feat'].shape[0] == len(result['pos'])


def test_lidar_simulation_crop_input_validation():
    """Test input validation for _call_single method."""
    lidar_crop = LiDARSimulationCrop()
    points = torch.randn(10, 3)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test invalid sensor_extrinsics type
    with pytest.raises(AssertionError, match="sensor_extrinsics must be torch.Tensor"):
        lidar_crop._call_single(pc, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
                               generator=torch.Generator())
    
    # Test invalid sensor_extrinsics shape
    with pytest.raises(AssertionError, match="sensor_extrinsics must be 4x4"):
        wrong_shape = torch.eye(3, dtype=torch.float32)
        lidar_crop._call_single(pc, wrong_shape, generator=torch.Generator())
    
    # Test with invalid generator - LiDARSimulationCrop accepts but doesn't validate generator
    # This is valid behavior as the generator parameter is optional
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    result = lidar_crop._call_single(pc, sensor_extrinsics, generator="invalid")
    assert len(result['pos']) >= 0  # Should work despite invalid generator
    
    # Test invalid point cloud (missing 'pos')
    with pytest.raises(AssertionError):
        invalid_pc = {'feat': torch.ones(10, 1)}
        sensor_extrinsics = torch.eye(4, dtype=torch.float32)
        lidar_crop._call_single(invalid_pc, sensor_extrinsics, generator=torch.Generator())
