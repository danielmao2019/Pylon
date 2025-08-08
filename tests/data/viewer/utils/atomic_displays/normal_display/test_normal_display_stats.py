"""Tests for normal display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

from data.viewer.utils.atomic_displays.normal_display import (
    get_normal_display_stats
)


# ================================================================================
# get_normal_display_stats Tests - Valid Cases
# ================================================================================

def test_get_normal_display_stats_basic(normal_tensor):
    """Test basic normal statistics with realistic normal vectors."""
    stats = get_normal_display_stats(normal_tensor)
    
    assert isinstance(stats, dict)
    assert 'shape' in stats
    assert 'dtype' in stats
    assert 'valid_pixels' in stats
    assert 'total_pixels' in stats
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'mean_magnitude' in stats
    
    # Basic validity checks
    assert stats['valid_pixels'] <= stats['total_pixels']
    assert stats['valid_pixels'] >= 0


def test_get_normal_display_stats_perfect_unit_normals():
    """Test normal statistics with perfect unit normal vectors."""
    # Create perfect unit normals
    normals = torch.zeros(3, 4, 4, dtype=torch.float32)
    normals[2, :, :] = 1.0  # All pointing up (0, 0, 1)
    
    stats = get_normal_display_stats(normals)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 16
    assert abs(stats['mean_magnitude'] - 1.0) < 1e-5  # Should be exactly 1.0


def test_get_normal_display_stats_mixed_normals():
    """Test normal statistics with mixed normal orientations."""
    normals = torch.zeros(3, 3, 3, dtype=torch.float32)
    
    # Set specific normals
    normals[:, 0, 0] = torch.tensor([1.0, 0.0, 0.0])  # X direction
    normals[:, 1, 1] = torch.tensor([0.0, 1.0, 0.0])  # Y direction  
    normals[:, 2, 2] = torch.tensor([0.0, 0.0, 1.0])  # Z direction
    
    # Normalize to unit vectors
    for i in range(3):
        for j in range(3):
            norm = torch.norm(normals[:, i, j])
            if norm > 0:
                normals[:, i, j] /= norm
    
    stats = get_normal_display_stats(normals)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 3  # Only 3 non-zero normals


def test_get_normal_display_stats_with_invalid_normals():
    """Test normal statistics with some invalid (inf, nan) normals."""
    normals = torch.randn(3, 16, 16, dtype=torch.float32)
    # Normalize to create valid unit normals
    magnitude = torch.sqrt((normals ** 2).sum(dim=0, keepdim=True))
    magnitude = torch.clamp(magnitude, min=1e-8)
    normals = normals / magnitude
    
    # Introduce some invalid normals
    normals[:, 0:3, 0:3] = float('inf')
    normals[:, 5:8, 5:8] = float('nan')
    
    stats = get_normal_display_stats(normals)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] < stats['total_pixels']


def test_get_normal_display_stats_all_invalid_normals():
    """Test normal statistics when all normals are invalid."""
    normals = torch.full((3, 32, 32), float('nan'), dtype=torch.float32)
    stats = get_normal_display_stats(normals)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 0
    assert stats['x_range'] == 'N/A'
    assert stats['y_range'] == 'N/A'
    assert stats['z_range'] == 'N/A'
    assert stats['mean_magnitude'] == 'N/A'


@pytest.mark.parametrize("tensor_size", [(16, 16), (32, 32), (64, 64)])
def test_get_normal_display_stats_various_sizes(tensor_size):
    """Test normal statistics with various tensor sizes."""
    h, w = tensor_size
    normals = torch.randn(3, h, w, dtype=torch.float32)
    # Normalize to unit vectors
    magnitude = torch.sqrt((normals ** 2).sum(dim=0, keepdim=True))
    magnitude = torch.clamp(magnitude, min=1e-8)
    normals = normals / magnitude
    
    stats = get_normal_display_stats(normals)
    
    assert isinstance(stats, dict)
    assert stats['total_pixels'] == h * w


def test_get_normal_display_stats_single_pixel():
    """Test normal statistics with single pixel normal."""
    normals = torch.tensor([[[0.0]], [[0.0]], [[1.0]]], dtype=torch.float32)
    stats = get_normal_display_stats(normals)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 1
    assert stats['total_pixels'] == 1
    assert abs(stats['mean_magnitude'] - 1.0) < 1e-5


# ================================================================================
# Batch Support Stats Tests - CRITICAL for eval viewer  
# ================================================================================

def test_get_normal_display_stats_batched(batched_normal_tensor):
    """Test normal statistics calculation for batched normal maps."""
    stats = get_normal_display_stats(batched_normal_tensor)
    
    assert isinstance(stats, dict)
    assert 'valid_pixels' in stats
    assert 'total_pixels' in stats


def test_batch_size_one_assertion_normal_stats():
    """Test that batch size > 1 raises assertion error in get_normal_display_stats."""
    invalid_batched_normals = torch.randn(3, 3, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_normal_display_stats(invalid_batched_normals)


# ================================================================================
# Integration Tests
# ================================================================================

def test_complete_batch_normal_stats_pipeline(batched_normal_tensor):
    """Test complete batched normal statistics pipeline."""
    stats = get_normal_display_stats(batched_normal_tensor)
    assert isinstance(stats, dict)
    assert 'valid_pixels' in stats