"""Tests for normal display statistics functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.normal_display import (
    get_normal_display_stats
)


# ================================================================================
# get_normal_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_normal_display_stats_invalid_input_type():
    """Test assertion failure for invalid normal input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_normal_display_stats_invalid_dimensions():
    """Test assertion failure for invalid dimensions."""
    # 2D tensor
    normals_2d = torch.rand(32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_2d)
    assert "Expected 3D [3,H,W] or 4D [N,3,H,W] tensor" in str(exc_info.value)
    
    # 1D tensor
    normals_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_1d)
    assert "Expected 3D [3,H,W] or 4D [N,3,H,W] tensor" in str(exc_info.value)


def test_get_normal_display_stats_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # Wrong number of channels for 3D tensor (should be 3 for normals)
    normals_wrong_channels = torch.rand(2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_wrong_channels)
    assert "Expected 3 channels" in str(exc_info.value)
    
    # Too many channels for 3D tensor
    normals_too_many = torch.rand(5, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_too_many)
    assert "Expected 3 channels" in str(exc_info.value)


def test_get_normal_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_normals = torch.empty((3, 0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(empty_normals)
    
    assert "cannot be empty" in str(exc_info.value)


def test_get_normal_display_stats_zero_dimensions():
    """Test assertion failure for zero-sized dimensions."""
    # Zero height
    normals_zero_h = torch.empty((3, 0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_zero_h)
    assert "cannot be empty" in str(exc_info.value)
    
    # Zero width  
    normals_zero_w = torch.empty((3, 32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_zero_w)
    assert "cannot be empty" in str(exc_info.value)