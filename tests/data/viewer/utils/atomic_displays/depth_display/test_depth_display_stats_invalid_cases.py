"""Tests for depth display statistics functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.depth_display import (
    get_depth_display_stats
)


# ================================================================================
# get_depth_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_depth_display_stats_invalid_input_type():
    """Test assertion failure for invalid depth input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_depth_display_stats_invalid_dimensions():
    """Test assertion failure for invalid dimensions."""
    # 1D tensor
    depth_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_1d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)
    
    # 4D tensor  
    depth_4d = torch.rand(2, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_4d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)


def test_get_depth_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_depth = torch.empty((0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(empty_depth)
    
    assert "cannot be empty" in str(exc_info.value)


def test_get_depth_display_stats_zero_dimensions():
    """Test assertion failure for zero-sized dimensions."""
    # Zero height
    depth_zero_h = torch.empty((0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_zero_h)
    assert "cannot be empty" in str(exc_info.value)
    
    # Zero width
    depth_zero_w = torch.empty((32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_zero_w)
    assert "cannot be empty" in str(exc_info.value)