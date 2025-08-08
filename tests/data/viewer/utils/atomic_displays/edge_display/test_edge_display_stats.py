"""Tests for edge display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

from data.viewer.utils.atomic_displays.edge_display import (
    get_edge_display_stats
)


# ================================================================================
# get_edge_display_stats Tests - Valid Cases
# ================================================================================

def test_get_edge_display_stats_3d_tensor(edge_tensor):
    """Test basic edge statistics with 3D tensor [1, H, W]."""
    stats = get_edge_display_stats(edge_tensor)
    
    assert isinstance(stats, dict)
    assert 'shape' in stats
    assert 'dtype' in stats
    assert 'total_pixels' in stats
    assert 'min_edge' in stats
    assert 'max_edge' in stats
    assert 'mean_edge' in stats
    assert 'std_edge' in stats
    assert 'edge_percentage' in stats
    
    # Basic validity checks  
    assert stats['valid_pixels'] <= stats['total_pixels']
    assert 0.0 <= stats['edge_percentage'] <= 100.0


def test_get_edge_display_stats_2d_tensor(edge_tensor_2d):
    """Test edge statistics with 2D tensor [H, W]."""
    stats = get_edge_display_stats(edge_tensor_2d)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] <= stats['total_pixels']


def test_get_edge_display_stats_binary_edges():
    """Test edge statistics with clear binary edge data."""
    # Create known binary edge pattern
    edges = torch.zeros(10, 10, dtype=torch.float32)
    edges[5, :] = 1.0  # Horizontal line
    edges[:, 5] = 1.0  # Vertical line
    edges = edges.unsqueeze(0)  # Add channel dimension [1, H, W]
    
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 100  # All pixels are valid for stats  
    assert stats['total_pixels'] == 100
    assert abs(stats['edge_percentage'] - 19.0) < 0.1


def test_get_edge_display_stats_no_edges():
    """Test edge statistics with no edges (all zeros)."""
    edges = torch.zeros(32, 32, dtype=torch.float32).unsqueeze(0)
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    assert stats['edge_percentage'] == 0.0


def test_get_edge_display_stats_all_edges():
    """Test edge statistics with all edges (all ones)."""
    edges = torch.ones(16, 16, dtype=torch.float32).unsqueeze(0)
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    assert stats['edge_percentage'] == 100.0


def test_get_edge_display_stats_with_invalid_values():
    """Test edge statistics with invalid values (negative, inf, nan)."""
    edges = torch.rand(32, 32, dtype=torch.float32)
    
    # Add some invalid values
    edges[0:3, 0:3] = float('inf')
    edges[5:8, 5:8] = float('nan')
    edges[10:13, 10:13] = -1.0
    
    edges = edges.unsqueeze(0)
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    # Should handle invalid values gracefully


def test_get_edge_display_stats_all_invalid():
    """Test edge statistics when all values are invalid."""
    edges = torch.full((1, 32, 32), float('nan'), dtype=torch.float32)
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    # Should return reasonable stats even with all invalid


@pytest.mark.parametrize("tensor_size", [(16, 16), (32, 32), (64, 64)])
def test_get_edge_display_stats_various_sizes(tensor_size):
    """Test edge statistics with various tensor sizes."""
    h, w = tensor_size
    edges = torch.rand(1, h, w, dtype=torch.float32)
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    assert stats['total_pixels'] == h * w


def test_get_edge_display_stats_single_pixel():
    """Test edge statistics with single pixel."""
    edges = torch.tensor([[[1.0]]], dtype=torch.float32)
    stats = get_edge_display_stats(edges)
    
    assert isinstance(stats, dict)
    assert stats['total_pixels'] == 1
    assert stats['edge_percentage'] == 100.0


def test_get_edge_display_stats_different_dtypes():
    """Test edge statistics with different tensor dtypes."""
    for dtype in [torch.float32, torch.float64]:
        edges = torch.rand(1, 16, 16, dtype=dtype)
        stats = get_edge_display_stats(edges)
        assert isinstance(stats, dict)


# ================================================================================
# Batch Support Stats Tests - CRITICAL for eval viewer  
# ================================================================================

def test_get_edge_display_stats_batched(batched_edge_tensor):
    """Test edge statistics calculation for batched edge maps."""
    stats = get_edge_display_stats(batched_edge_tensor)
    
    assert isinstance(stats, dict)
    assert 'edge_percentage' in stats
    assert 'total_pixels' in stats


def test_batch_size_one_assertion_edge_stats():
    """Test that batch size > 1 raises assertion error in get_edge_display_stats."""
    invalid_batched_edges = torch.rand(3, 32, 32, dtype=torch.float32)  # [N, H, W] with N=3
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_edge_display_stats(invalid_batched_edges)


# ================================================================================
# Integration Tests
# ================================================================================

def test_complete_batch_edge_stats_pipeline(batched_edge_tensor):
    """Test complete batched edge statistics pipeline."""
    stats = get_edge_display_stats(batched_edge_tensor)
    assert isinstance(stats, dict)
    assert 'edge_percentage' in stats
