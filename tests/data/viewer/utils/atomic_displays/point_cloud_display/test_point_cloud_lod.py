"""Tests for point cloud Level-of-Detail (LOD) functionality.

Focuses specifically on the apply_lod_to_point_cloud function.
CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.point_cloud_display import (
    apply_lod_to_point_cloud
)


# ================================================================================
# apply_lod_to_point_cloud Tests
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