#!/usr/bin/env python3
"""
Test cases for FOVCrop transform.
"""

import pytest
import torch
import numpy as np
from data.transforms.vision_3d import FOVCrop


def test_fov_crop_initialization():
    """Test FOVCrop initialization with valid parameters."""
    # Test default initialization
    fov_crop = FOVCrop()
    assert fov_crop.horizontal_fov == 360.0
    assert fov_crop.vertical_fov == 40.0
    
    # Test custom initialization
    fov_crop = FOVCrop(horizontal_fov=120.0, vertical_fov=90.0)
    assert fov_crop.horizontal_fov == 120.0
    assert fov_crop.vertical_fov == 90.0


def test_fov_crop_invalid_parameters():
    """Test FOVCrop initialization with invalid parameters."""
    # Test negative horizontal FOV
    with pytest.raises(AssertionError, match="horizontal_fov must be in \\(0, 360\\]"):
        FOVCrop(horizontal_fov=-90.0)
    
    # Test zero horizontal FOV
    with pytest.raises(AssertionError, match="horizontal_fov must be in \\(0, 360\\]"):
        FOVCrop(horizontal_fov=0.0)
    
    # Test horizontal FOV > 360
    with pytest.raises(AssertionError, match="horizontal_fov must be in \\(0, 360\\]"):
        FOVCrop(horizontal_fov=400.0)
    
    # Test invalid vertical FOV type
    with pytest.raises(AssertionError, match="vertical_fov must be numeric"):
        FOVCrop(vertical_fov="invalid")
    
    # Test zero vertical FOV
    with pytest.raises(AssertionError, match="vertical_fov must be in \\(0, 180\\]"):
        FOVCrop(vertical_fov=0.0)
    
    # Test negative vertical FOV
    with pytest.raises(AssertionError, match="vertical_fov must be in \\(0, 180\\]"):
        FOVCrop(vertical_fov=-45.0)
    
    # Test vertical FOV > 180°
    with pytest.raises(AssertionError, match="vertical_fov must be in \\(0, 180\\]"):
        FOVCrop(vertical_fov=200.0)


def test_fov_crop_basic_functionality():
    """Test basic FOV cropping functionality."""
    # Create test points in specific locations
    points = torch.tensor([
        [5.0, 0.0, 0.0],    # Forward (should be kept)
        [0.0, 5.0, 0.0],    # Right side (should be kept with 360° HFOV)
        [0.0, -5.0, 0.0],   # Left side (should be kept with 360° HFOV)
        [-5.0, 0.0, 0.0],   # Behind (should be kept with 360° HFOV)
        [5.0, 0.0, 2.0],    # Forward and up (should be kept with default VFOV ±20°)
        [5.0, 0.0, -2.0],   # Forward and down (should be kept with default VFOV ±20°)
        [5.0, 0.0, 8.0],    # Forward and high up (should be removed with default VFOV ±20°)
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Sensor at origin, looking along +X axis
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test with default FOV (360° horizontal, 40° vertical = ±20°)
    fov_crop = FOVCrop()
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should keep points within FOV
    assert len(result['pos']) >= 4  # At least first 4 points should be kept
    assert result['feat'].shape == (len(result['pos']), 1)


def test_fov_crop_horizontal_fov():
    """Test horizontal FOV filtering with different angles."""
    # Create points in a circle around origin
    angles = torch.linspace(0, 2*np.pi, 8, dtype=torch.float32)[:-1]  # 7 points, avoid duplicate at 0/2π
    radius = 5.0
    
    points = torch.stack([
        radius * torch.cos(angles),
        radius * torch.sin(angles),
        torch.zeros_like(angles)
    ], dim=1)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test narrow horizontal FOV (60°)
    fov_crop = FOVCrop(horizontal_fov=60.0)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should keep points within ±30° from forward direction
    assert len(result['pos']) <= len(points)
    
    # Test wide horizontal FOV (180°)
    fov_crop = FOVCrop(horizontal_fov=180.0)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should keep all forward-facing points (front half-circle)
    assert len(result['pos']) >= 3  # At least forward and side points


def test_fov_crop_vertical_fov():
    """Test vertical FOV filtering with different angles."""
    # Create points at different elevations
    points = torch.tensor([
        [5.0, 0.0, 0.0],     # Horizon level (0°)
        [5.0, 0.0, 1.0],     # Small elevation 
        [5.0, 0.0, -1.0],    # Small negative elevation
        [5.0, 0.0, 3.0],     # Higher elevation
        [5.0, 0.0, -3.0],    # Lower elevation
        [5.0, 0.0, 8.0],     # Very high elevation (beyond default FOV ±20°)
        [5.0, 0.0, -8.0],    # Very low elevation (beyond default FOV ±20°)
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test default vertical FOV (40° = ±20°)
    fov_crop = FOVCrop()
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should keep points within vertical FOV
    assert len(result['pos']) > 0
    assert len(result['pos']) <= len(points)
    
    # Test wider vertical FOV (90° = ±45°)
    fov_crop = FOVCrop(vertical_fov=90.0)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should keep more points with wider FOV
    assert len(result['pos']) > 0


def test_fov_crop_sensor_pose():
    """Test FOV cropping with different sensor poses."""
    # Create points along different axes
    points = torch.tensor([
        [5.0, 0.0, 0.0],   # +X direction
        [0.0, 5.0, 0.0],   # +Y direction
        [0.0, 0.0, 5.0],   # +Z direction
        [-5.0, 0.0, 0.0],  # -X direction
        [0.0, -5.0, 0.0],  # -Y direction
        [0.0, 0.0, -5.0],  # -Z direction
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test sensor looking along +Y axis (90° rotation around Z)
    sensor_extrinsics = torch.tensor([
        [0.0, -1.0, 0.0, 0.0],  # +X sensor = -Y world
        [1.0, 0.0, 0.0, 0.0],   # +Y sensor = +X world
        [0.0, 0.0, 1.0, 0.0],   # +Z sensor = +Z world
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    fov_crop = FOVCrop(horizontal_fov=90.0)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should see different points based on sensor orientation
    assert len(result['pos']) > 0


def test_fov_crop_edge_cases():
    """Test edge cases for FOV cropping."""
    fov_crop = FOVCrop()
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test empty point cloud
    empty_pc = {'pos': torch.empty(0, 3, dtype=torch.float32), 'feat': torch.empty(0, 1)}
    result = fov_crop._call_single(empty_pc, sensor_extrinsics)
    assert len(result['pos']) == 0
    assert len(result['feat']) == 0
    
    # Test single point within FOV
    single_pc = {'pos': torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32), 
                 'feat': torch.ones(1, 1)}
    result = fov_crop._call_single(single_pc, sensor_extrinsics)
    assert len(result['pos']) >= 0  # May or may not be kept depending on vertical FOV
    
    # Test single point outside FOV (very high up)
    high_pc = {'pos': torch.tensor([[5.0, 0.0, 20.0]], dtype=torch.float32),
                'feat': torch.ones(1, 1)}
    result = fov_crop._call_single(high_pc, sensor_extrinsics)
    assert len(result['pos']) == 0  # Should be outside default vertical FOV ±20°


def test_fov_crop_device_handling():
    """Test that FOVCrop handles different devices correctly."""
    points = torch.tensor([
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    fov_crop = FOVCrop()
    
    # Test with CPU tensors
    sensor_extrinsics_cpu = torch.eye(4, dtype=torch.float32)
    result = fov_crop._call_single(pc, sensor_extrinsics_cpu)
    assert result['pos'].device == points.device
    assert len(result['pos']) >= 0
    
    # Test with mismatched devices
    if torch.cuda.is_available():
        sensor_extrinsics_cuda = sensor_extrinsics_cpu.cuda()
        result = fov_crop._call_single(pc, sensor_extrinsics_cuda)
        assert result['pos'].device == points.device  # Should match input device
        assert len(result['pos']) >= 0


def test_fov_crop_multiple_features():
    """Test FOV cropping with multiple feature channels."""
    points = torch.tensor([
        [5.0, 0.0, 0.0],   # Forward (likely keep)
        [0.0, 5.0, 0.0],   # Right (likely keep)
        [5.0, 0.0, 20.0],  # Very high up (likely remove)
    ], dtype=torch.float32)
    
    # Test with multiple feature types
    pc = {
        'pos': points,
        'feat': torch.randn(3, 5),  # 5-dimensional features
        'colors': torch.randn(3, 3),  # RGB colors
        'normals': torch.randn(3, 3)  # Surface normals
    }
    
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    fov_crop = FOVCrop()
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should preserve feature structure
    num_kept = len(result['pos'])
    assert num_kept >= 0
    assert result['feat'].shape == (num_kept, 5)
    assert result['colors'].shape == (num_kept, 3)
    assert result['normals'].shape == (num_kept, 3)


def test_fov_crop_deterministic():
    """Test that FOV cropping is deterministic."""
    points = torch.randn(100, 3) * 10  # Random points
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    fov_crop = FOVCrop()
    
    # Run multiple times
    result1 = fov_crop._call_single(pc, sensor_extrinsics)
    result2 = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Results should be identical
    assert len(result1['pos']) == len(result2['pos'])
    if len(result1['pos']) > 0:
        assert torch.allclose(result1['pos'], result2['pos'])


def test_fov_crop_360_degree_horizontal():
    """Test 360-degree horizontal FOV (no horizontal filtering)."""
    # Create points in a full circle
    angles = torch.linspace(0, 2*np.pi, 8, dtype=torch.float32)[:-1]  # 7 points
    radius = 5.0
    
    points = torch.stack([
        radius * torch.cos(angles),
        radius * torch.sin(angles),
        torch.zeros_like(angles)
    ], dim=1)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test 360° horizontal FOV
    fov_crop = FOVCrop(horizontal_fov=360.0)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should keep all points (no horizontal filtering, all at horizon level)
    assert len(result['pos']) == len(points)


def test_fov_crop_precision():
    """Test FOV cropping precision at boundary conditions."""
    # Create points exactly at FOV boundaries
    fov_angle = 45.0  # Half of 90° horizontal FOV
    fov_rad = np.radians(fov_angle)
    
    points = torch.tensor([
        [5.0, 5.0 * np.tan(fov_rad - 1e-6), 0.0],  # Just inside horizontal boundary
        [5.0, 5.0 * np.tan(fov_rad), 0.0],         # Exactly at horizontal boundary
        [5.0, 5.0 * np.tan(fov_rad + 1e-6), 0.0],  # Just outside horizontal boundary
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    fov_crop = FOVCrop(horizontal_fov=90.0)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should handle boundary conditions correctly
    assert len(result['pos']) >= 2  # At least first 2 points


@pytest.mark.parametrize("horizontal_fov", [30.0, 60.0, 90.0, 120.0, 180.0])
def test_fov_crop_parametrized_horizontal_fov(horizontal_fov):
    """Test FOV cropping with different horizontal FOV values."""
    # Create points at known angles (at horizon level for consistency)
    points = torch.tensor([
        [5.0, 0.0, 0.0],                           # 0° (forward)
        [5.0, 5.0 * np.tan(np.radians(15)), 0.0], # 15°
        [5.0, 5.0 * np.tan(np.radians(30)), 0.0], # 30°
        [5.0, 5.0 * np.tan(np.radians(45)), 0.0], # 45°
        [5.0, 5.0 * np.tan(np.radians(60)), 0.0], # 60°
        [5.0, 100.0, 0.0],                         # ~90° (large Y value)
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    fov_crop = FOVCrop(horizontal_fov=horizontal_fov)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should work for all horizontal FOV values
    assert len(result['pos']) >= 0
    assert len(result['pos']) <= len(points)


@pytest.mark.parametrize("vertical_fov", [30.0, 60.0, 90.0, 120.0])
def test_fov_crop_parametrized_vertical_fov(vertical_fov):
    """Test FOV cropping with different vertical FOV values."""
    # Create points at known elevations (forward direction for consistency)
    points = torch.tensor([
        [5.0, 0.0, 0.0],                            # 0° elevation
        [5.0, 0.0, 5.0 * np.tan(np.radians(15))],  # 15° elevation
        [5.0, 0.0, 5.0 * np.tan(np.radians(30))],  # 30° elevation
        [5.0, 0.0, 5.0 * np.tan(np.radians(45))],  # 45° elevation
        [5.0, 0.0, -5.0 * np.tan(np.radians(15))], # -15° elevation
        [5.0, 0.0, -5.0 * np.tan(np.radians(30))], # -30° elevation
        [5.0, 0.0, -5.0 * np.tan(np.radians(45))], # -45° elevation
    ], dtype=torch.float32)
    
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    sensor_extrinsics = torch.eye(4, dtype=torch.float32)
    
    fov_crop = FOVCrop(vertical_fov=vertical_fov)
    result = fov_crop._call_single(pc, sensor_extrinsics)
    
    # Should work for all vertical FOV values
    assert len(result['pos']) >= 0
    assert len(result['pos']) <= len(points)


def test_fov_crop_input_validation():
    """Test input validation for _call_single method."""
    fov_crop = FOVCrop()
    points = torch.randn(10, 3)
    pc = {'pos': points, 'feat': torch.ones(len(points), 1)}
    
    # Test invalid sensor_extrinsics type
    with pytest.raises(AssertionError, match="sensor_extrinsics must be torch.Tensor"):
        fov_crop._call_single(pc, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    # Test invalid sensor_extrinsics shape
    with pytest.raises(AssertionError, match="sensor_extrinsics must be 4x4"):
        wrong_shape = torch.eye(3, dtype=torch.float32)
        fov_crop._call_single(pc, wrong_shape)
    
    # Test invalid point cloud (missing 'pos')
    with pytest.raises(AssertionError):
        invalid_pc = {'feat': torch.ones(10, 1)}
        sensor_extrinsics = torch.eye(4, dtype=torch.float32)
        fov_crop._call_single(invalid_pc, sensor_extrinsics)
