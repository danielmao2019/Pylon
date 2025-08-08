"""Tests for instance surrogate display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.instance_surrogate_display import (
    create_instance_surrogate_display,
    get_instance_surrogate_display_stats,
    _convert_surrogate_to_instance_mask
)


# ================================================================================
# create_instance_surrogate_display Tests - Invalid Cases
# ================================================================================

def test_create_instance_surrogate_display_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display("not_a_tensor", "Test")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_create_instance_surrogate_display_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    instance_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_1d, "Test")
    assert "Expected 3D [2,H,W] or 4D [N,2,H,W] tensor" in str(exc_info.value)
    
    # 2D tensor
    instance_2d = torch.randn(32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_2d, "Test")
    assert "Expected 3D [2,H,W] or 4D [N,2,H,W] tensor" in str(exc_info.value)
    
    # 4D tensor with invalid batch size
    instance_4d = torch.randn(2, 2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_4d, "Test")
    assert "Expected batch size 1 for visualization, got 2" in str(exc_info.value)


def test_create_instance_surrogate_display_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 1 channel
    instance_1ch = torch.randn(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_1ch, "Test")
    assert "Expected 2 channels [2, H, W]" in str(exc_info.value)
    
    # 3 channels
    instance_3ch = torch.randn(3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_3ch, "Test")
    assert "Expected 2 channels [2, H, W]" in str(exc_info.value)
    
    # 4 channels
    instance_4ch = torch.randn(4, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_4ch, "Test")
    assert "Expected 2 channels [2, H, W]" in str(exc_info.value)


def test_create_instance_surrogate_display_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_instance = torch.empty((2, 0, 0), dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(empty_instance, "Test")
    
    assert "Instance surrogate tensor cannot be empty" in str(exc_info.value)


def test_create_instance_surrogate_display_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height
    instance_zero_h = torch.empty((2, 0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_zero_h, "Test")
    assert "Instance surrogate tensor cannot be empty" in str(exc_info.value)
    
    # Zero width
    instance_zero_w = torch.empty((2, 32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_zero_w, "Test")
    assert "Instance surrogate tensor cannot be empty" in str(exc_info.value)


def test_create_instance_surrogate_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_surrogate, 123)
    
    assert "Expected str title" in str(exc_info.value)


def test_create_instance_surrogate_display_invalid_ignore_value_type():
    """Test assertion failure for invalid ignore_value type."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        create_instance_surrogate_display(instance_surrogate, "Test", ignore_value="not_int")
    
    assert "Expected int ignore_value" in str(exc_info.value)


# ================================================================================
# get_instance_surrogate_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_instance_surrogate_display_stats_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    instance_1d = torch.randn(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_1d)
    assert "Expected 3D [2,H,W] or 4D [N,2,H,W] tensor" in str(exc_info.value)
    
    # 2D tensor
    instance_2d = torch.randn(32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_2d)
    assert "Expected 3D [2,H,W] or 4D [N,2,H,W] tensor" in str(exc_info.value)
    
    # 4D tensor with invalid batch size
    instance_4d = torch.randn(2, 2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_4d)
    assert "Expected batch size 1 for analysis, got 2" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_invalid_channels():
    """Test assertion failure for wrong number of channels."""
    # 1 channel
    instance_1ch = torch.randn(1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_1ch)
    assert "Expected 2 channels [2, H, W]" in str(exc_info.value)
    
    # 3 channels
    instance_3ch = torch.randn(3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_3ch)
    assert "Expected 2 channels [2, H, W]" in str(exc_info.value)
    
    # 4 channels
    instance_4ch = torch.randn(4, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_4ch)
    assert "Expected 2 channels [2, H, W]" in str(exc_info.value)


def test_get_instance_surrogate_display_stats_invalid_ignore_index_type():
    """Test assertion failure for invalid ignore_index type."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_instance_surrogate_display_stats(instance_surrogate, ignore_index="not_int")
    
    assert "Expected int ignore_index" in str(exc_info.value)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_create_instance_surrogate_display_with_different_dtypes():
    """Test instance surrogate display with various tensor dtypes (should work)."""
    # Float64
    instance_f64 = torch.randn(2, 32, 32, dtype=torch.float64) * 5.0
    fig = create_instance_surrogate_display(instance_f64, "Float64 Test")
    assert isinstance(fig, go.Figure)
    
    # Instance surrogates should be float32, not float16 or integer


def test_get_instance_surrogate_display_stats_with_different_dtypes():
    """Test statistics with various tensor dtypes (should work)."""
    # Float64
    instance_f64 = torch.randn(2, 32, 32, dtype=torch.float64) * 5.0
    stats = get_instance_surrogate_display_stats(instance_f64)
    assert isinstance(stats, dict)
    assert "torch.float64" in stats["Data Type"]
    
    # Instance surrogates should be float, not integer


def test_instance_surrogate_display_with_extreme_tensor_shapes():
    """Test with extreme but valid tensor shapes."""
    # Very thin tensor
    thin_instance = torch.randn(2, 1, 1000, dtype=torch.float32) * 5.0
    fig = create_instance_surrogate_display(thin_instance, "Thin Instance")
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(thin_instance)
    assert stats["Height"] == 1
    assert stats["Width"] == 1000
    
    # Very tall tensor
    tall_instance = torch.randn(2, 1000, 1, dtype=torch.float32) * 5.0
    fig = create_instance_surrogate_display(tall_instance, "Tall Instance")
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(tall_instance)
    assert stats["Height"] == 1000
    assert stats["Width"] == 1


def test_instance_surrogate_display_single_pixel():
    """Test instance surrogate display with single pixel."""
    # Single pixel
    single_pixel = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.float32)
    
    fig = create_instance_surrogate_display(single_pixel, "Single Pixel")
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(single_pixel)
    assert stats["Height"] == 1
    assert stats["Width"] == 1
    assert stats["Total Pixels"] == 1
    assert stats["Valid Pixels"] == 1
    assert stats["Ignore Pixels"] == 0


def test_instance_surrogate_display_with_unusual_values():
    """Test instance surrogate display with unusual but valid values."""
    # All zeros (no offsets)
    zero_instance = torch.zeros(2, 32, 32, dtype=torch.float32)
    fig = create_instance_surrogate_display(zero_instance, "Zero Offsets")
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(zero_instance)
    assert stats["Valid Pixels"] == 32 * 32  # All should be valid
    assert float(stats["Y Offset Mean"]) == 0.0
    assert float(stats["X Offset Mean"]) == 0.0
    assert float(stats["Magnitude Mean"]) == 0.0
    
    # Very large offsets
    large_instance = torch.full((2, 32, 32), 1000.0, dtype=torch.float32)
    fig = create_instance_surrogate_display(large_instance, "Large Offsets")
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(large_instance)
    assert stats["Valid Pixels"] == 32 * 32
    assert float(stats["Y Offset Mean"]) == 1000.0
    assert float(stats["X Offset Mean"]) == 1000.0


def test_instance_surrogate_display_with_mixed_valid_invalid_ignore_values():
    """Test instance surrogate display with mixed valid and invalid ignore scenarios."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    # Only Y channel has ignore values (unusual case)
    instance_surrogate[0, :10, :10] = 250  # Y channel ignore
    # X channel remains valid - this creates inconsistent ignore regions
    
    # This should still work (though it's an unusual case)
    fig = create_instance_surrogate_display(instance_surrogate, "Mixed Ignore")
    assert isinstance(fig, go.Figure)
    
    # For statistics, ignore mask is created with AND condition
    stats = get_instance_surrogate_display_stats(instance_surrogate)
    assert isinstance(stats, dict)
    # Since only Y channel has ignore value, these pixels won't be considered ignore
    assert stats["Valid Pixels"] == 32 * 32  # All pixels considered valid


def test_instance_surrogate_display_boundary_ignore_values():
    """Test instance surrogate display with boundary ignore value scenarios."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    # Use ignore value that's close to actual data range
    instance_surrogate[:, 10:20, 10:20] = 10.0  # Set region to value 10
    
    # Use ignore_value=10 (same as some data)
    fig = create_instance_surrogate_display(instance_surrogate, "Boundary Ignore", ignore_value=10)
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(instance_surrogate, ignore_index=10)
    # Pixels with value exactly 10 in BOTH channels should be considered ignore
    assert isinstance(stats, dict)


def test_convert_surrogate_to_instance_mask_edge_cases():
    """Test _convert_surrogate_to_instance_mask with edge cases."""
    # Very small tensors
    y_tiny = torch.tensor([[1.0]], dtype=torch.float32)
    x_tiny = torch.tensor([[2.0]], dtype=torch.float32)
    mask_tiny = _convert_surrogate_to_instance_mask(y_tiny, x_tiny, ignore_index=250)
    assert mask_tiny.shape == (1, 1)
    assert mask_tiny.dtype == torch.int64
    
    # All same values (edge case for quantile calculation)
    y_same = torch.full((32, 32), 5.0, dtype=torch.float32)
    x_same = torch.full((32, 32), 5.0, dtype=torch.float32)
    mask_same = _convert_surrogate_to_instance_mask(y_same, x_same, ignore_index=250)
    assert mask_same.shape == (32, 32)
    assert mask_same.dtype == torch.int64
    
    # Mix of finite and infinite values
    y_mixed = torch.randn(32, 32, dtype=torch.float32)
    x_mixed = torch.randn(32, 32, dtype=torch.float32)
    y_mixed[:5, :5] = float('inf')  # Some infinite values
    x_mixed[:5, :5] = float('inf')
    
    # Should handle inf values gracefully
    mask_mixed = _convert_surrogate_to_instance_mask(y_mixed, x_mixed, ignore_index=250)
    assert mask_mixed.shape == (32, 32)
    assert mask_mixed.dtype == torch.int64


def test_instance_surrogate_display_negative_ignore_values():
    """Test instance surrogate display with negative ignore values."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    # Use negative ignore value
    instance_surrogate[:, :10, :10] = -100
    
    fig = create_instance_surrogate_display(instance_surrogate, "Negative Ignore", ignore_value=-100)
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(instance_surrogate, ignore_index=-100)
    assert isinstance(stats, dict)
    assert stats["Ignore Pixels"] > 0  # Should have some ignore pixels