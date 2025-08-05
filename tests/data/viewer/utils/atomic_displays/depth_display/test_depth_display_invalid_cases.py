"""Tests for depth display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.depth_display import (
    create_depth_display,
    get_depth_display_stats
)


# ================================================================================
# create_depth_display Tests - Invalid Cases
# ================================================================================

def test_create_depth_display_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display("not_a_tensor", "Test")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_create_depth_display_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    depth_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth_1d, "Test")
    assert "Expected 2D tensor [H,W]" in str(exc_info.value)
    
    # 3D tensor
    depth_3d = torch.rand(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth_3d, "Test")
    assert "Expected 2D tensor [H,W]" in str(exc_info.value)
    
    # 4D tensor
    depth_4d = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth_4d, "Test")
    assert "Expected 2D tensor [H,W]" in str(exc_info.value)


def test_create_depth_display_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_depth = torch.empty((0, 0), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(empty_depth, "Test")
    
    assert "Depth tensor cannot be empty" in str(exc_info.value)


def test_create_depth_display_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height
    depth_zero_h = torch.empty((0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth_zero_h, "Test")
    assert "Depth tensor cannot be empty" in str(exc_info.value)
    
    # Zero width
    depth_zero_w = torch.empty((32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth_zero_w, "Test")
    assert "Depth tensor cannot be empty" in str(exc_info.value)


def test_create_depth_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    depth = torch.rand(32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth, 123)
    
    assert "Expected str title" in str(exc_info.value)


def test_create_depth_display_invalid_colorscale_type():
    """Test assertion failure for invalid colorscale type."""
    depth = torch.rand(32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_depth_display(depth, "Test", colorscale=123)
    
    assert "Expected str colorscale" in str(exc_info.value)


# ================================================================================
# get_depth_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_depth_display_stats_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_depth_display_stats_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    depth_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_1d)
    assert "Expected 2D tensor [H,W]" in str(exc_info.value)
    
    # 3D tensor
    depth_3d = torch.rand(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_3d)
    assert "Expected 2D tensor [H,W]" in str(exc_info.value)
    
    # 4D tensor
    depth_4d = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_4d)
    assert "Expected 2D tensor [H,W]" in str(exc_info.value)


def test_get_depth_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_depth = torch.empty((0, 0), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(empty_depth)
    
    assert "Depth tensor cannot be empty" in str(exc_info.value)


def test_get_depth_display_stats_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height
    depth_zero_h = torch.empty((0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_zero_h)
    assert "Depth tensor cannot be empty" in str(exc_info.value)
    
    # Zero width
    depth_zero_w = torch.empty((32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_depth_display_stats(depth_zero_w)
    assert "Depth tensor cannot be empty" in str(exc_info.value)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_create_depth_display_with_different_dtypes():
    """Test depth display with various tensor dtypes (should work)."""
    # Float64
    depth_f64 = torch.rand(32, 32, dtype=torch.float64) * 10.0 + 0.1
    fig = create_depth_display(depth_f64, "Float64 Test")
    assert isinstance(fig, go.Figure)
    
    # Float16
    depth_f16 = torch.rand(32, 32, dtype=torch.float16) * 10.0 + 0.1
    fig = create_depth_display(depth_f16, "Float16 Test")
    assert isinstance(fig, go.Figure)


def test_get_depth_display_stats_with_different_dtypes():
    """Test statistics with various float tensor dtypes."""
    # Float64 (high precision depth)
    depth_f64 = torch.rand(32, 32, dtype=torch.float64) * 10.0 + 0.1
    stats = get_depth_display_stats(depth_f64)
    assert isinstance(stats, dict)
    assert "torch.float64" in stats["dtype"]


def test_depth_display_with_extreme_tensor_shapes():
    """Test with extreme but valid tensor shapes."""
    # Very thin tensor
    thin_depth = torch.rand(1, 1000, dtype=torch.float32) * 10.0 + 0.1
    fig = create_depth_display(thin_depth, "Thin Depth")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(thin_depth)
    assert stats["shape"] == [1, 1000]
    
    # Very tall tensor
    tall_depth = torch.rand(1000, 1, dtype=torch.float32) * 10.0 + 0.1
    fig = create_depth_display(tall_depth, "Tall Depth")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(tall_depth)
    assert stats["shape"] == [1000, 1]


def test_depth_display_with_unusual_values():
    """Test depth display with unusual but valid values."""
    # All zeros (invalid depths, but should not crash)
    zero_depth = torch.zeros(32, 32, dtype=torch.float32)
    fig = create_depth_display(zero_depth, "Zero Depths")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(zero_depth)
    assert stats["valid_pixels"] == 0  # All zeros are invalid
    assert stats["min_depth"] == "N/A"
    assert stats["max_depth"] == "N/A"
    
    # Negative depths (invalid, but should not crash)
    negative_depth = torch.full((32, 32), -5.0, dtype=torch.float32)
    fig = create_depth_display(negative_depth, "Negative Depths")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(negative_depth)
    assert stats["valid_pixels"] == 0  # All negative depths are invalid
    
    # Very large depths (should work)
    large_depth = torch.full((32, 32), 1e6, dtype=torch.float32)
    fig = create_depth_display(large_depth, "Large Depths")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(large_depth)
    assert stats["valid_pixels"] == 32 * 32  # All should be valid
    assert stats["min_depth"] == 1e6
    assert stats["max_depth"] == 1e6


def test_depth_display_with_mixed_valid_invalid_values():
    """Test depth display with mix of valid and invalid values."""
    depth = torch.rand(32, 32, dtype=torch.float32) * 10.0 + 0.1
    
    # Set some regions to invalid values
    depth[:10, :10] = 0.0         # Zero depths (invalid)
    depth[10:20, 10:20] = -1.0    # Negative depths (invalid)
    depth[20:30, 20:30] = float('nan')  # NaN depths (invalid)
    
    # This should still work
    fig = create_depth_display(depth, "Mixed Valid/Invalid")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(depth)
    assert isinstance(stats, dict)
    assert stats["valid_pixels"] < stats["total_pixels"]


def test_depth_display_single_pixel_edge_cases():
    """Test depth display with single pixel edge cases."""
    # Single valid pixel
    valid_single = torch.tensor([[5.0]], dtype=torch.float32)
    fig = create_depth_display(valid_single, "Single Valid")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(valid_single)
    assert stats["valid_pixels"] == 1
    assert stats["min_depth"] == 5.0
    assert stats["max_depth"] == 5.0
    
    # Single invalid pixel (zero)
    invalid_single = torch.tensor([[0.0]], dtype=torch.float32)
    fig = create_depth_display(invalid_single, "Single Invalid")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(invalid_single)
    assert stats["valid_pixels"] == 0
    assert stats["min_depth"] == "N/A"
    
    # Single invalid pixel (NaN)
    nan_single = torch.tensor([[float('nan')]], dtype=torch.float32)
    fig = create_depth_display(nan_single, "Single NaN")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(nan_single)
    assert stats["valid_pixels"] == 0
    assert stats["min_depth"] == "N/A"


def test_depth_display_boundary_values():
    """Test depth display with boundary depth values."""
    # Very small positive depths (just above zero)
    tiny_depth = torch.full((32, 32), 1e-10, dtype=torch.float32)
    fig = create_depth_display(tiny_depth, "Tiny Depths")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(tiny_depth)
    assert stats["valid_pixels"] == 32 * 32  # Should be valid (positive)
    assert abs(stats["min_depth"] - 1e-10) < 1e-15  # Use approximate comparison for float precision
    
    # Just at zero boundary
    zero_boundary = torch.zeros(32, 32, dtype=torch.float32)
    zero_boundary[16, 16] = 1e-15  # One tiny positive value
    
    fig = create_depth_display(zero_boundary, "Zero Boundary")
    assert isinstance(fig, go.Figure)
    
    stats = get_depth_display_stats(zero_boundary)
    assert stats["valid_pixels"] == 1  # Only one valid pixel