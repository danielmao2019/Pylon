"""Tests for instance surrogate display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

from data.viewer.utils.atomic_displays.instance_surrogate_display import (
    get_instance_surrogate_display_stats
)


# ================================================================================
# get_instance_surrogate_display_stats Tests - Valid Cases
# ================================================================================

def test_get_instance_surrogate_display_stats_basic(instance_surrogate_tensor):
    """Test basic instance surrogate statistics."""
    stats = get_instance_surrogate_display_stats(instance_surrogate_tensor)
    
    assert isinstance(stats, dict)
    assert 'shape' in stats
    assert 'dtype' in stats
    assert 'total_pixels' in stats
    assert 'valid_pixels' in stats
    assert 'ignore_pixels' in stats
    assert 'y_offset_range' in stats
    assert 'x_offset_range' in stats
    
    # Basic validity checks
    assert stats['valid_pixels'] + stats['ignore_pixels'] <= stats['total_pixels']
    assert stats['valid_pixels'] >= 0
    assert stats['ignore_pixels'] >= 0


def test_get_instance_surrogate_display_stats_custom_ignore_value():
    """Test instance surrogate statistics with custom ignore value."""
    # Create surrogate with custom ignore value
    surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    # Set some pixels to custom ignore value
    custom_ignore = 999.0
    surrogate[:, 0:5, 0:5] = custom_ignore
    
    stats = get_instance_surrogate_display_stats(surrogate, ignore_value=custom_ignore)
    
    assert isinstance(stats, dict)
    assert stats['ignore_pixels'] == 25  # 5x5 region


def test_get_instance_surrogate_display_stats_no_valid_pixels():
    """Test instance surrogate statistics with no valid pixels (all ignore)."""
    # Create surrogate where all pixels are ignore value
    surrogate = torch.full((2, 16, 16), 250, dtype=torch.float32)
    stats = get_instance_surrogate_display_stats(surrogate)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 0
    assert stats['ignore_pixels'] == 256  # 16x16
    assert stats['total_pixels'] == 256


def test_get_instance_surrogate_display_stats_all_valid_pixels():
    """Test instance surrogate statistics with all valid pixels (no ignore)."""
    # Create surrogate with known values, no ignore pixels
    surrogate = torch.zeros(2, 8, 8, dtype=torch.float32)
    surrogate[0] = 2.0  # Y offsets
    surrogate[1] = -1.5  # X offsets
    
    stats = get_instance_surrogate_display_stats(surrogate)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 64  # 8x8
    assert stats['ignore_pixels'] == 0
    assert abs(float(stats['y_offset_range'].split()[0][1:]) - 2.0) < 1e-5  # Min Y offset
    assert abs(float(stats['x_offset_range'].split()[0][1:]) - (-1.5)) < 1e-5  # Min X offset


def test_get_instance_surrogate_display_stats_known_values():
    """Test instance surrogate statistics with known offset values."""
    surrogate = torch.zeros(2, 4, 4, dtype=torch.float32)
    
    # Set known Y offsets (first channel)
    surrogate[0, 0, :] = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    # Set known X offsets (second channel)  
    surrogate[1, 0, :] = torch.tensor([-3.0, -1.5, 0.5, 3.0])
    
    # Set some ignore regions
    surrogate[:, 1:, :] = 250  # Ignore all but first row
    
    stats = get_instance_surrogate_display_stats(surrogate)
    
    assert isinstance(stats, dict)
    assert stats['valid_pixels'] == 4  # Only first row
    assert stats['ignore_pixels'] == 12  # Remaining pixels


@pytest.mark.parametrize("tensor_size", [(16, 16), (32, 32), (64, 64)])
def test_get_instance_surrogate_display_stats_various_sizes(tensor_size):
    """Test instance surrogate statistics with various tensor sizes."""
    h, w = tensor_size
    surrogate = torch.randn(2, h, w, dtype=torch.float32) * 10.0
    stats = get_instance_surrogate_display_stats(surrogate)
    
    assert isinstance(stats, dict)
    assert stats['total_pixels'] == h * w


def test_get_instance_surrogate_display_stats_different_dtypes():
    """Test instance surrogate statistics with different tensor dtypes."""
    for dtype in [torch.float32, torch.float64]:
        surrogate = torch.randn(2, 16, 16, dtype=dtype) * 3.0
        stats = get_instance_surrogate_display_stats(surrogate)
        assert isinstance(stats, dict)


# ================================================================================
# Batch Support Stats Tests - CRITICAL for eval viewer  
# ================================================================================

def test_get_instance_surrogate_display_stats_batched(batched_instance_surrogate_tensor):
    """Test instance surrogate statistics calculation for batched data."""
    stats = get_instance_surrogate_display_stats(batched_instance_surrogate_tensor)
    
    assert isinstance(stats, dict)
    assert 'valid_pixels' in stats
    assert 'total_pixels' in stats


def test_batch_size_one_assertion_instance_surrogate_stats():
    """Test that batch size > 1 raises assertion error in get_instance_surrogate_display_stats."""
    invalid_batched_surrogate = torch.randn(3, 2, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_instance_surrogate_display_stats(invalid_batched_surrogate)


# ================================================================================
# Integration Tests
# ================================================================================

def test_complete_batch_instance_surrogate_stats_pipeline(batched_instance_surrogate_tensor):
    """Test complete batched instance surrogate statistics pipeline."""
    stats = get_instance_surrogate_display_stats(batched_instance_surrogate_tensor)
    assert isinstance(stats, dict)
    assert 'valid_pixels' in stats