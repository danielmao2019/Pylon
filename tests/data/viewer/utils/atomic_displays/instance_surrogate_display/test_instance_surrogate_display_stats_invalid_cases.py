"""Tests for instance surrogate display statistics functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.instance_surrogate_display import (
    get_instance_surrogate_display_stats
)


# ================================================================================
# get_instance_surrogate_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_instance_surrogate_display_stats_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_invalid_dimensions():
    """Test assertion failure for invalid dimensions."""
    # 2D tensor
    surrogate_2d = torch.rand(32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(surrogate_2d)
    assert "Expected 3D [2,H,W] or 4D [N,2,H,W] tensor" in str(exc_info.value)
    
    # 1D tensor
    surrogate_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(surrogate_1d)
    assert "Expected 3D [2,H,W] or 4D [N,2,H,W] tensor" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # Wrong number of channels for 3D tensor (should be 2 for Y/X offsets)
    surrogate_wrong_channels = torch.rand(3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(surrogate_wrong_channels)
    assert "Expected 2 channels" in str(exc_info.value)
    
    # Single channel
    surrogate_single = torch.rand(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(surrogate_single)
    assert "Expected 2 channels" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_invalid_ignore_index_type():
    """Test assertion failure for invalid ignore_index type."""
    surrogate = torch.randn(2, 16, 16, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(surrogate, ignore_index="invalid")
    
    assert "Expected int ignore_index" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_surrogate = torch.empty((2, 0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(empty_surrogate)
    
    assert "cannot be empty" in str(exc_info.value)
