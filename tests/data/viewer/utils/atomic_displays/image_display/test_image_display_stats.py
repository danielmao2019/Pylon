"""Tests for image display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional

from data.viewer.utils.atomic_displays.image_display import (
    get_image_display_stats
)


# ================================================================================
# get_image_display_stats Tests - Valid Cases
# ================================================================================

def test_get_image_display_stats_basic():
    """Test basic image statistics calculation."""
    image = torch.randn(3, 32, 32, dtype=torch.float32)
    stats = get_image_display_stats(image)
    
    assert isinstance(stats, dict)
    assert "Shape" in stats
    assert "Min Value" in stats
    assert "Max Value" in stats
    assert "Mean Value" in stats
    assert "Std Dev" in stats
    
    assert stats["Shape"] == "(3, 32, 32)"


def test_image_stats_with_edge_cases():
    """Test image statistics with edge case tensors."""
    # All zeros
    zero_image = torch.zeros(3, 32, 32, dtype=torch.float32)
    stats = get_image_display_stats(zero_image)
    assert float(stats["Min Value"]) == 0.0
    assert float(stats["Max Value"]) == 0.0
    
    # Single pixel
    tiny_image = torch.ones(3, 1, 1, dtype=torch.float32)
    stats = get_image_display_stats(tiny_image)
    assert stats["Shape"] == "(3, 1, 1)"


# ================================================================================
# Batch Support Stats Tests - CRITICAL for eval viewer
# ================================================================================

def test_get_image_display_stats_batched_rgb(batched_rgb_tensor):
    """Test image statistics calculation for batched RGB image."""
    stats = get_image_display_stats(batched_rgb_tensor)
    
    assert isinstance(stats, dict)
    assert "Shape" in stats
    assert "Min Value" in stats
    assert "Max Value" in stats
    assert "Mean Value" in stats
    assert "Std Dev" in stats
    
    # Should show unbatched shape in stats
    assert stats["Shape"] == "(3, 32, 32)"


def test_batch_size_one_assertion_stats():
    """Test that batch size > 1 raises assertion error in get_image_display_stats."""
    invalid_batched_image = torch.randint(0, 255, (3, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_image_display_stats(invalid_batched_image)


# ================================================================================
# Integration Tests
# ================================================================================

def test_complete_batch_stats_pipeline_rgb(batched_rgb_tensor):
    """Test complete batched image statistics pipeline."""
    stats = get_image_display_stats(batched_rgb_tensor)
    assert isinstance(stats, dict)
    assert len(stats) >= 5
    
    # Stats should show unbatched shape
    assert stats["Shape"] == "(3, 32, 32)"


def test_complete_batch_stats_pipeline_grayscale(batched_grayscale_tensor):
    """Test complete batched grayscale statistics pipeline."""
    stats = get_image_display_stats(batched_grayscale_tensor)
    
    assert isinstance(stats, dict)
    assert stats["Shape"] == "(1, 32, 32)"
