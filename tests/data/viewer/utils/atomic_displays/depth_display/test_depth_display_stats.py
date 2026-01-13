"""Tests for depth display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

from data.viewer.utils.atomic_displays.depth_display import (
    get_depth_display_stats
)


# ================================================================================
# get_depth_display_stats Tests - Valid Cases
# ================================================================================

def test_get_depth_display_stats_basic(depth_tensor):
    """Test basic depth statistics calculation."""
    stats = get_depth_display_stats(depth_tensor)

    assert isinstance(stats, dict)
    assert 'shape' in stats
    assert 'dtype' in stats
    assert 'valid_pixels' in stats
    assert 'total_pixels' in stats
    assert 'min_depth' in stats
    assert 'max_depth' in stats
    assert 'mean_depth' in stats
    assert 'std_depth' in stats

    # Basic validity checks
    assert stats['valid_pixels'] <= stats['total_pixels']
    assert stats['valid_pixels'] >= 0


def test_get_depth_display_stats_realistic_depths():
    """Test depth statistics with realistic depth values."""
    depth_map = torch.rand(64, 64, dtype=torch.float32) * 10.0 + 0.1  # Range [0.1, 10.1]
    stats = get_depth_display_stats(depth_map)

    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 64 * 64  # All should be valid
    assert float(stats['min_depth']) >= 0.1
    assert float(stats['max_depth']) <= 10.1


def test_get_depth_display_stats_with_invalid_depths():
    """Test depth statistics with some invalid (negative, zero, inf, nan) depths."""
    depth_map = torch.rand(32, 32, dtype=torch.float32) * 5.0 + 1.0

    # Introduce some invalid depths
    depth_map[0:5, 0:5] = 0.0      # Zero depths
    depth_map[0:3, 10:15] = -1.0   # Negative depths
    depth_map[10:12, 0:3] = float('inf')  # Infinite depths
    depth_map[20:22, 20:22] = float('nan')  # NaN depths

    stats = get_depth_display_stats(depth_map)

    assert isinstance(stats, dict)
    assert stats['valid_pixels'] < stats['total_pixels']  # Some should be invalid
    assert stats['valid_pixels'] >= 0


def test_get_depth_display_stats_all_invalid():
    """Test depth statistics when all depths are invalid."""
    depth_map = torch.full((32, 32), -1.0, dtype=torch.float32)  # All negative
    stats = get_depth_display_stats(depth_map)

    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 0
    assert stats['total_pixels'] == 32 * 32


def test_get_depth_display_stats_zero_negative_depths():
    """Test depth statistics with zero and negative depth values."""
    depth_map = torch.tensor([
        [0.0, -1.0, 2.0],
        [1.5, 0.0, 3.0],
        [-0.5, 2.5, 1.0]
    ], dtype=torch.float32)

    stats = get_depth_display_stats(depth_map)

    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 5  # Only positive values (2.0, 1.5, 3.0, 2.5, 1.0)
    assert stats['total_pixels'] == 9


@pytest.mark.parametrize("tensor_size", [(16, 16), (32, 32), (64, 64), (128, 128)])
def test_get_depth_display_stats_various_sizes(tensor_size):
    """Test depth statistics with various tensor sizes."""
    h, w = tensor_size
    depth_map = torch.rand(h, w, dtype=torch.float32) * 5.0 + 0.5
    stats = get_depth_display_stats(depth_map)

    assert isinstance(stats, dict)
    assert stats['total_pixels'] == h * w


def test_get_depth_display_stats_single_pixel():
    """Test depth statistics with single pixel depth map."""
    depth_map = torch.tensor([[2.5]], dtype=torch.float32)
    stats = get_depth_display_stats(depth_map)

    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 1
    assert stats['total_pixels'] == 1
    assert float(stats['min_depth']) == 2.5
    assert float(stats['max_depth']) == 2.5


def test_get_depth_display_stats_different_dtypes():
    """Test depth statistics with different tensor dtypes."""
    for dtype in [torch.float32, torch.float64]:
        depth_map = torch.rand(16, 16, dtype=dtype) * 3.0 + 1.0
        stats = get_depth_display_stats(depth_map)

        assert isinstance(stats, dict)
        assert stats['valid_pixels'] == 16 * 16


# ================================================================================
# Batch Support Stats Tests - CRITICAL for eval viewer
# ================================================================================

def test_get_depth_display_stats_batched(batched_depth_tensor):
    """Test depth statistics calculation for batched depth maps."""
    stats = get_depth_display_stats(batched_depth_tensor)

    assert isinstance(stats, dict)
    assert 'valid_pixels' in stats
    assert 'total_pixels' in stats


def test_batch_size_one_assertion_depth_stats():
    """Test that batch size > 1 raises assertion error in get_depth_display_stats."""
    invalid_batched_depth = torch.rand(3, 32, 32, dtype=torch.float32)

    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_depth_display_stats(invalid_batched_depth)


# ================================================================================
# Integration Tests
# ================================================================================

def test_complete_batch_depth_stats_pipeline(batched_depth_tensor):
    """Test complete batched depth statistics pipeline."""
    stats = get_depth_display_stats(batched_depth_tensor)
    assert isinstance(stats, dict)
    assert 'valid_pixels' in stats