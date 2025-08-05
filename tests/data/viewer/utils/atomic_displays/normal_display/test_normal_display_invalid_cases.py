"""Tests for normal display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import plotly.graph_objects as go
import numpy as np

from data.viewer.utils.atomic_displays.normal_display import (
    create_normal_display,
    get_normal_display_stats
)


# ================================================================================
# create_normal_display Tests - Invalid Cases
# ================================================================================

def test_create_normal_display_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display("not_a_tensor", "Test")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_create_normal_display_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 2D tensor
    normals_2d = torch.randn(32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_2d, "Test")
    assert "Expected 3D tensor [3,H,W]" in str(exc_info.value)
    
    # 4D tensor
    normals_4d = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_4d, "Test")
    assert "Expected 3D tensor [3,H,W]" in str(exc_info.value)
    
    # 1D tensor
    normals_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_1d, "Test")
    assert "Expected 3D tensor [3,H,W]" in str(exc_info.value)


def test_create_normal_display_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 1 channel
    normals_1ch = torch.randn(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_1ch, "Test")
    assert "Expected 3 channels for normals" in str(exc_info.value)
    
    # 4 channels
    normals_4ch = torch.randn(4, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_4ch, "Test")
    assert "Expected 3 channels for normals" in str(exc_info.value)
    
    # 2 channels
    normals_2ch = torch.randn(2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_2ch, "Test")
    assert "Expected 3 channels for normals" in str(exc_info.value)


def test_create_normal_display_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_normals = torch.empty((3, 0, 0), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(empty_normals, "Test")
    
    assert "Normal tensor cannot be empty" in str(exc_info.value)


def test_create_normal_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    normals = torch.randn(3, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals, 123)
    
    assert "Expected str title" in str(exc_info.value)


def test_create_normal_display_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height
    normals_zero_h = torch.empty((3, 0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_zero_h, "Test")
    assert "Normal tensor cannot be empty" in str(exc_info.value)
    
    # Zero width
    normals_zero_w = torch.empty((3, 32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_normal_display(normals_zero_w, "Test")
    assert "Normal tensor cannot be empty" in str(exc_info.value)


# ================================================================================
# get_normal_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_normal_display_stats_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_normal_display_stats_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 2D tensor
    normals_2d = torch.randn(32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_2d)
    assert "Expected 3D tensor [3,H,W]" in str(exc_info.value)
    
    # 4D tensor
    normals_4d = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_4d)
    assert "Expected 3D tensor [3,H,W]" in str(exc_info.value)
    
    # 1D tensor
    normals_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_1d)
    assert "Expected 3D tensor [3,H,W]" in str(exc_info.value)


def test_get_normal_display_stats_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 1 channel
    normals_1ch = torch.randn(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_1ch)
    assert "Expected 3 channels" in str(exc_info.value)
    
    # 4 channels
    normals_4ch = torch.randn(4, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_4ch)
    assert "Expected 3 channels" in str(exc_info.value)
    
    # 2 channels
    normals_2ch = torch.randn(2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_2ch)
    assert "Expected 3 channels" in str(exc_info.value)


def test_get_normal_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_normals = torch.empty((3, 0, 0), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(empty_normals)
    
    assert "Normal tensor cannot be empty" in str(exc_info.value)


def test_get_normal_display_stats_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height
    normals_zero_h = torch.empty((3, 0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_zero_h)
    assert "Normal tensor cannot be empty" in str(exc_info.value)
    
    # Zero width
    normals_zero_w = torch.empty((3, 32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_normal_display_stats(normals_zero_w)
    assert "Normal tensor cannot be empty" in str(exc_info.value)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_create_normal_display_with_different_dtypes():
    """Test normal display with different tensor dtypes (should work)."""
    # These should work despite not being float32
    normals_float64 = torch.randn(3, 32, 32, dtype=torch.float64)
    fig = create_normal_display(normals_float64, "Float64 Test")
    assert isinstance(fig, go.Figure)
    
    # Normal vectors should be float32, not float16


def test_get_normal_display_stats_with_different_dtypes():
    """Test statistics with different tensor dtypes (should work)."""
    # Float64
    normals_float64 = torch.randn(3, 32, 32, dtype=torch.float64)
    stats = get_normal_display_stats(normals_float64)
    assert isinstance(stats, dict)
    assert "torch.float64" in stats["dtype"]
    
    # Normal vectors should be float32, not float16


def test_normal_display_with_float32_dtypes():
    """Test normal display with proper float32 dtypes."""
    # Normal vectors should be float, not integer (unit vectors)
    normals_f32 = torch.randn(3, 32, 32, dtype=torch.float32)
    fig = create_normal_display(normals_f32, "Float32 Normals")
    assert isinstance(fig, go.Figure)
    
    stats = get_normal_display_stats(normals_f32)
    assert isinstance(stats, dict)


def test_normal_display_with_extreme_tensor_shapes():
    """Test with extreme but valid tensor shapes."""
    # Very thin tensor
    thin_normals = torch.randn(3, 1, 1000, dtype=torch.float32)
    fig = create_normal_display(thin_normals, "Thin Normals")
    assert isinstance(fig, go.Figure)
    
    stats = get_normal_display_stats(thin_normals)
    assert stats["shape"] == [3, 1, 1000]
    
    # Very tall tensor
    tall_normals = torch.randn(3, 1000, 1, dtype=torch.float32)
    fig = create_normal_display(tall_normals, "Tall Normals")
    assert isinstance(fig, go.Figure)
    
    stats = get_normal_display_stats(tall_normals)
    assert stats["shape"] == [3, 1000, 1]


def test_normal_display_with_all_zero_normals():
    """Test normal display with all zero normal vectors."""
    # All zero normals (degenerate case)
    zero_normals = torch.zeros(3, 32, 32, dtype=torch.float32)
    
    fig = create_normal_display(zero_normals, "Zero Normals")
    assert isinstance(fig, go.Figure)
    
    stats = get_normal_display_stats(zero_normals)
    assert isinstance(stats, dict)
    assert stats["mean_magnitude"] == 0.0


def test_normal_display_with_mixed_valid_invalid_normals():
    """Test normal display with mix of valid and invalid values."""
    normals = torch.randn(3, 32, 32, dtype=torch.float32)
    
    # Set some regions to NaN
    normals[:, :10, :10] = float('nan')
    
    # Set some regions to infinity
    normals[:, 10:20, 10:20] = float('inf')
    
    # This should still work
    fig = create_normal_display(normals, "Mixed Valid/Invalid")
    assert isinstance(fig, go.Figure)
    
    stats = get_normal_display_stats(normals)
    assert isinstance(stats, dict)
    assert stats["valid_pixels"] < stats["total_pixels"]