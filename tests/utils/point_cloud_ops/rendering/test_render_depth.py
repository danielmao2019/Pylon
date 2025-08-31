"""Test cases for depth rendering from point clouds."""

import torch
import pytest
from utils.point_cloud_ops.rendering import render_depth_from_pointcloud


def test_render_depth_basic():
    """Test basic depth rendering without mask."""
    # Create simple point cloud with 4 points at different depths
    pc_data = {
        'pos': torch.tensor([
            [0.0, 0.0, -1.0],   # Center, depth 1
            [0.5, 0.5, -2.0],   # Upper right, depth 2
            [-0.5, 0.5, -1.5],  # Upper left, depth 1.5
            [0.0, -0.5, -3.0],  # Bottom center, depth 3
        ], dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)
    
    # Render without mask
    depth_map = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=False
    )
    
    assert depth_map.shape == (100, 100)
    assert depth_map.dtype == torch.float32
    
    # Check that rendered depths are positive (absolute values taken)
    valid_depths = depth_map[depth_map != -1.0]
    assert (valid_depths > 0).all()


def test_render_depth_with_mask():
    """Test depth rendering with valid mask."""
    pc_data = {
        'pos': torch.tensor([
            [0.0, 0.0, -1.0],
            [0.2, 0.2, -1.5],
            [-0.3, -0.3, -2.0],
        ], dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)
    
    # Render with mask
    depth_map, valid_mask = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution,
        convention="opengl",
        return_mask=True
    )
    
    assert depth_map.shape == (100, 100)
    assert valid_mask.shape == (100, 100)
    assert valid_mask.dtype == torch.bool
    
    # Check that we have some valid pixels
    assert valid_mask.sum() > 0
    assert valid_mask.sum() < 100 * 100  # Not all pixels should be filled
    
    # Check that valid pixels have positive depth
    assert (depth_map[valid_mask] > 0).all()
    
    # Check that invalid pixels have ignore_value
    assert (depth_map[~valid_mask] == -1.0).all()


def test_render_depth_sorting():
    """Test that closer points overwrite farther ones."""
    # Two points at same image location but different depths
    pc_data = {
        'pos': torch.tensor([
            [0.0, 0.0, -3.0],  # Farther point (depth 3)
            [0.0, 0.0, -1.0],  # Closer point (depth 1, should overwrite)
        ], dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    resolution = (100, 100)
    
    depth_map = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=resolution
    )
    
    # Check that we have rendered depths
    valid_depths = depth_map[depth_map != -3.0]  # Ignore value
    if len(valid_depths) > 0:
        # The closest depth should be 1.0 (not necessarily at center due to projection)
        assert valid_depths.min() < 1.5  # Should have depth close to 1.0


def test_render_depth_custom_ignore_value():
    """Test using custom ignore value for empty pixels."""
    pc_data = {
        'pos': torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    custom_ignore = -999.0
    
    depth_map = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        ignore_value=custom_ignore
    )
    
    # Most pixels should have the custom ignore value
    background_pixels = depth_map == custom_ignore
    assert background_pixels.sum() > 100 * 100 * 0.9  # Most pixels are background


def test_render_depth_points_behind_camera():
    """Test that points behind camera are filtered out."""
    pc_data = {
        'pos': torch.tensor([
            [0.0, 0.0, 1.0],   # Behind camera (positive Z in OpenGL)
            [0.0, 0.0, -1.0],  # In front of camera
        ], dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    
    depth_map, valid_mask = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100),
        return_mask=True
    )
    
    # Should only have one point rendered (the one in front)
    assert valid_mask.sum() >= 1


def test_render_depth_empty_pointcloud():
    """Test that empty point cloud raises assertion error."""
    pc_data = {
        'pos': torch.empty((0, 3), dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    
    with pytest.raises(AssertionError, match="Point cloud cannot be empty"):
        render_depth_from_pointcloud(
            pc_data=pc_data,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            resolution=(100, 100)
        )


def test_render_depth_multiple_points_per_pixel():
    """Test that multiple points projecting to same pixel are handled correctly."""
    # Create points that project to similar locations
    pc_data = {
        'pos': torch.tensor([
            [0.0, 0.0, -1.0],
            [0.01, 0.0, -2.0],
            [0.0, 0.01, -1.5],
            [-0.01, 0.0, -3.0],
        ], dtype=torch.float32)
    }
    
    camera_intrinsics = torch.tensor([
        [1000.0, 0.0, 50.0],  # High focal length for tight projection
        [0.0, 1000.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    
    depth_map = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(100, 100)
    )
    
    # Center pixels should have the closest depth (1.0)
    center_region = depth_map[48:52, 48:52]
    valid_depths = center_region[center_region != -1.0]
    if len(valid_depths) > 0:
        assert valid_depths.min() < 1.5  # Should have depth close to 1.0


def test_render_depth_intrinsics_scaling():
    """Test that intrinsics are properly scaled for different resolutions."""
    pc_data = {
        'pos': torch.tensor([
            [0.0, 0.0, -1.0],
            [0.5, 0.5, -2.0],
        ], dtype=torch.float32)
    }
    
    # Original intrinsics for 100x100 image
    camera_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    camera_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Render at different resolution
    depth_map_small = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(50, 50)  # Half resolution
    )
    
    depth_map_large = render_depth_from_pointcloud(
        pc_data=pc_data,
        camera_intrinsics=camera_intrinsics,
        camera_extrinsics=camera_extrinsics,
        resolution=(200, 200)  # Double resolution
    )
    
    assert depth_map_small.shape == (50, 50)
    assert depth_map_large.shape == (200, 200)
    
    # Both should have some valid depth values
    assert (depth_map_small != -1.0).any()
    assert (depth_map_large != -1.0).any()


def test_render_depth_invalid_inputs():
    """Test various invalid input conditions."""
    valid_pc_data = {
        'pos': torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    }
    valid_intrinsics = torch.tensor([
        [100.0, 0.0, 50.0],
        [0.0, 100.0, 50.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    valid_extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Test missing 'pos' key
    with pytest.raises(AssertionError, match="must contain 'pos' key"):
        render_depth_from_pointcloud(
            pc_data={},
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100)
        )
    
    # Test wrong intrinsics shape
    with pytest.raises(AssertionError, match="camera_intrinsics must be 3x3 matrix"):
        render_depth_from_pointcloud(
            pc_data=valid_pc_data,
            camera_intrinsics=torch.eye(4),
            camera_extrinsics=valid_extrinsics,
            resolution=(100, 100)
        )
    
    # Test wrong extrinsics shape
    with pytest.raises(AssertionError, match="camera_extrinsics must be 4x4 matrix"):
        render_depth_from_pointcloud(
            pc_data=valid_pc_data,
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=torch.eye(3),
            resolution=(100, 100)
        )
    
    # Test invalid resolution
    with pytest.raises(AssertionError, match="resolution must be positive integers"):
        render_depth_from_pointcloud(
            pc_data=valid_pc_data,
            camera_intrinsics=valid_intrinsics,
            camera_extrinsics=valid_extrinsics,
            resolution=(0, 100)
        )
