"""Tests for point cloud display statistics functionality.

Focuses specifically on the get_point_cloud_display_stats function.
CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.point_cloud_display import (
    get_point_cloud_display_stats
)


# ================================================================================
# get_point_cloud_display_stats Tests
# ================================================================================

def test_get_point_cloud_display_stats_basic(point_cloud_3d):
    """Test basic point cloud statistics calculation."""
    pc_dict = {'pos': point_cloud_3d}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 1000
    assert stats['dimensions'] == 3
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'center' in stats
    assert len(stats['x_range']) == 2
    assert len(stats['y_range']) == 2  
    assert len(stats['z_range']) == 2
    assert len(stats['center']) == 3


def test_get_point_cloud_display_stats_known_values():
    """Test statistics with known point cloud values."""
    # Create point cloud with known properties
    points = torch.zeros(100, 3, dtype=torch.float32)
    
    # Set specific coordinate ranges
    points[:25, 0] = 1.0    # X coordinates: 25 points at x=1
    points[25:50, 0] = 2.0  # X coordinates: 25 points at x=2
    points[50:75, 1] = 3.0  # Y coordinates: 25 points at y=3
    points[75:, 2] = 4.0    # Z coordinates: 25 points at z=4
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['x_range'] == [0.0, 2.0]  # X range
    assert stats['y_range'] == [0.0, 3.0]  # Y range
    assert stats['z_range'] == [0.0, 4.0]  # Z range


def test_get_point_cloud_display_stats_single_point():
    """Test statistics with single point."""
    single_point = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)
    
    pc_dict = {'pos': single_point}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 1
    assert stats['x_range'] == [1.5, 1.5]  # X range
    assert stats['y_range'] == [2.5, 2.5]  # Y range  
    assert stats['z_range'] == [3.5, 3.5]  # Z range


def test_get_point_cloud_display_stats_uniform_distribution():
    """Test statistics with uniformly distributed points."""
    # Create points in unit cube [0,1]^3
    points = torch.rand(1000, 3, dtype=torch.float32)
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 1000
    assert stats['dimensions'] == 3
    # Ranges should be approximately [0, 1] for each dimension (allowing some randomness tolerance)
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert len(stats['x_range']) == 2
    assert len(stats['y_range']) == 2
    assert len(stats['z_range']) == 2


@pytest.mark.parametrize("n_points", [10, 100, 1000, 5000])
def test_get_point_cloud_display_stats_various_sizes(n_points):
    """Test statistics with various point cloud sizes."""
    points = torch.randn(n_points, 3, dtype=torch.float32)
    
    pc_dict = {'pos': points}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == n_points


def test_get_point_cloud_display_stats_different_dtypes():
    """Test statistics with different tensor dtypes."""
    # Float32
    points_f32 = torch.randn(100, 3, dtype=torch.float32)
    pc_dict_f32 = {'pos': points_f32}
    stats_f32 = get_point_cloud_display_stats(pc_dict_f32)
    assert isinstance(stats_f32, dict)
    
    # Float64
    points_f64 = torch.randn(100, 3, dtype=torch.float64)
    pc_dict_f64 = {'pos': points_f64}
    stats_f64 = get_point_cloud_display_stats(pc_dict_f64)
    assert isinstance(stats_f64, dict)
    
    # Integer (unusual but should work)
    points_int = torch.randint(-10, 10, (100, 3), dtype=torch.int32)
    pc_dict_int = {'pos': points_int}
    stats_int = get_point_cloud_display_stats(pc_dict_int)
    assert isinstance(stats_int, dict)


def test_get_point_cloud_display_stats_extreme_coordinates():
    """Test statistics with extreme coordinate values."""
    # Very large coordinates
    large_points = torch.full((100, 3), 1e6, dtype=torch.float32)
    pc_dict_large = {'pos': large_points}
    stats_large = get_point_cloud_display_stats(pc_dict_large)
    assert isinstance(stats_large, dict)
    assert stats_large['total_points'] == 100
    
    # Very small coordinates
    small_points = torch.full((100, 3), 1e-6, dtype=torch.float32)
    pc_dict_small = {'pos': small_points}
    stats_small = get_point_cloud_display_stats(pc_dict_small)
    assert isinstance(stats_small, dict)
    
    # Mixed positive/negative
    mixed_points = torch.randn(100, 3, dtype=torch.float32) * 1000
    pc_dict_mixed = {'pos': mixed_points}
    stats_mixed = get_point_cloud_display_stats(pc_dict_mixed)
    assert isinstance(stats_mixed, dict)


def test_get_point_cloud_display_stats_with_4_channels():
    """Test that get_point_cloud_display_stats accepts 4+ channels (valid case)."""
    # 4 channels should be VALID - function accepts [N, 3+] 
    pc_4ch = torch.randn(100, 4, dtype=torch.float32)
    pc_dict = {'pos': pc_4ch}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 100
    assert stats['dimensions'] == 4  # Should show 4 dimensions


def test_get_point_cloud_display_stats_with_6_channels():
    """Test that get_point_cloud_display_stats accepts 6 channels (valid case)."""
    # 6 channels should be VALID - function accepts [N, 3+]
    pc_6ch = torch.randn(50, 6, dtype=torch.float32)
    pc_dict = {'pos': pc_6ch}
    stats = get_point_cloud_display_stats(pc_dict)
    
    assert isinstance(stats, dict)
    assert stats['total_points'] == 50
    assert stats['dimensions'] == 6  # Should show 6 dimensions


def test_get_point_cloud_display_stats_available_fields():
    """Test that available_fields are correctly reported."""
    # Test with just positions
    pc_dict_pos_only = {'pos': torch.randn(100, 3, dtype=torch.float32)}
    stats = get_point_cloud_display_stats(pc_dict_pos_only)
    assert 'available_fields' in stats
    assert stats['available_fields'] == ['pos']
    
    # Test with positions and colors
    pc_dict_with_rgb = {
        'pos': torch.randn(100, 3, dtype=torch.float32),
        'rgb': torch.randn(100, 3, dtype=torch.float32)
    }
    stats_rgb = get_point_cloud_display_stats(pc_dict_with_rgb)
    assert 'available_fields' in stats_rgb
    assert set(stats_rgb['available_fields']) == {'pos', 'rgb'}
    
    # Test with multiple fields
    pc_dict_multi = {
        'pos': torch.randn(100, 3, dtype=torch.float32),
        'rgb': torch.randn(100, 3, dtype=torch.float32),
        'normals': torch.randn(100, 3, dtype=torch.float32),
        'features': torch.randn(100, 64, dtype=torch.float32)
    }
    stats_multi = get_point_cloud_display_stats(pc_dict_multi)
    assert 'available_fields' in stats_multi
    assert set(stats_multi['available_fields']) == {'pos', 'rgb', 'normals', 'features'}
