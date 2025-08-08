"""Tests for edge display statistics functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.edge_display import (
    get_edge_display_stats
)


# ================================================================================
# get_edge_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_edge_display_stats_invalid_input_type():
    """Test assertion failure for invalid edge input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_edge_display_stats_invalid_dimensions():
    """Test assertion failure for invalid dimensions."""
    # 1D tensor
    edges_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_1d)
    assert "Expected 2D [H,W] or 3D" in str(exc_info.value)
    
    # 4D tensor
    edges_4d = torch.rand(2, 2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_4d)
    assert "Expected 2D [H,W] or 3D" in str(exc_info.value)


def test_get_edge_display_stats_invalid_channels_3d():
    """Test assertion failure for 3D tensor with wrong number of channels."""
    # Wrong number of channels for 3D tensor
    edges_wrong_channels = torch.rand(3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_wrong_channels)
    assert "Expected batch size 1 for analysis" in str(exc_info.value)


def test_get_edge_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_edges = torch.empty((0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(empty_edges)
    
    assert "cannot be empty" in str(exc_info.value)


def test_get_edge_display_stats_zero_dimensions():
    """Test assertion failure for zero-sized dimensions."""
    # Zero height
    edges_zero_h = torch.empty((0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_zero_h)
    assert "cannot be empty" in str(exc_info.value)
    
    # Zero width  
    edges_zero_w = torch.empty((32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_zero_w)
    assert "cannot be empty" in str(exc_info.value)