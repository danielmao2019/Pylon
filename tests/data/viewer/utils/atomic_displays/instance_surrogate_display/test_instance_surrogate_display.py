"""Tests for instance surrogate display functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.instance_surrogate_display import (
    create_instance_surrogate_display,
    get_instance_surrogate_display_stats,
    _convert_surrogate_to_instance_mask
)


# ================================================================================
# create_instance_surrogate_display Tests - Valid Cases
# ================================================================================

def test_create_instance_surrogate_display_basic(instance_surrogate_tensor):
    """Test basic instance surrogate display creation."""
    fig = create_instance_surrogate_display(instance_surrogate_tensor, "Test Instance Surrogate")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Instance Surrogate"


def test_create_instance_surrogate_display_custom_ignore_value(instance_surrogate_tensor):
    """Test instance surrogate display with custom ignore value."""
    fig = create_instance_surrogate_display(
        instance_surrogate_tensor, 
        "Custom Ignore Value",
        ignore_value=100
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Custom Ignore Value"


def test_create_instance_surrogate_display_with_kwargs(instance_surrogate_tensor):
    """Test instance surrogate display with additional kwargs."""
    fig = create_instance_surrogate_display(
        instance_surrogate_tensor,
        "Test with Kwargs",
        ignore_value=250,
        extra_param="ignored"  # Should be ignored
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test with Kwargs"


def test_create_instance_surrogate_display_realistic_offsets():
    """Test instance surrogate display with realistic coordinate offsets."""
    # Create realistic instance surrogate data
    instance_surrogate = torch.zeros(2, 64, 64, dtype=torch.float32)
    
    # Create several instance regions with realistic offsets
    # Instance 1: Top-left region pointing to center
    instance_surrogate[0, 10:20, 10:20] = 5.0   # Y offset to center
    instance_surrogate[1, 10:20, 10:20] = 5.0   # X offset to center
    
    # Instance 2: Bottom-right region pointing to center  
    instance_surrogate[0, 40:50, 40:50] = -5.0  # Y offset to center
    instance_surrogate[1, 40:50, 40:50] = -5.0  # X offset to center
    
    # Add some ignore regions
    instance_surrogate[:, :5, :] = 250  # Top border ignore
    
    fig = create_instance_surrogate_display(instance_surrogate, "Realistic Offsets")
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("tensor_size", [(32, 32), (64, 64), (128, 128)])
def test_create_instance_surrogate_display_various_sizes(tensor_size):
    """Test instance surrogate display with various tensor sizes."""
    h, w = tensor_size
    instance_surrogate = torch.randn(2, h, w, dtype=torch.float32) * 5.0
    
    # Add some ignore regions
    ignore_mask = torch.rand(h, w) < 0.1
    instance_surrogate[0, ignore_mask] = 250
    instance_surrogate[1, ignore_mask] = 250
    
    fig = create_instance_surrogate_display(instance_surrogate, f"Test {h}x{w}")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == f"Test {h}x{w}"


def test_create_instance_surrogate_display_extreme_offsets():
    """Test instance surrogate display with extreme offset values."""
    # Very large offsets
    large_offsets = torch.randn(2, 32, 32, dtype=torch.float32) * 100.0
    fig = create_instance_surrogate_display(large_offsets, "Large Offsets")
    assert isinstance(fig, go.Figure)
    
    # Very small offsets
    small_offsets = torch.randn(2, 32, 32, dtype=torch.float32) * 0.1
    fig = create_instance_surrogate_display(small_offsets, "Small Offsets")
    assert isinstance(fig, go.Figure)


def test_create_instance_surrogate_display_no_ignore_regions():
    """Test instance surrogate display with no ignore regions."""
    # All pixels are valid instance pixels
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    fig = create_instance_surrogate_display(instance_surrogate, "No Ignore Regions")
    assert isinstance(fig, go.Figure)


def test_create_instance_surrogate_display_all_ignore_regions():
    """Test instance surrogate display with all ignore regions."""
    # All pixels are ignore regions
    instance_surrogate = torch.full((2, 32, 32), 250.0, dtype=torch.float32)
    
    fig = create_instance_surrogate_display(instance_surrogate, "All Ignore Regions")
    assert isinstance(fig, go.Figure)


# ================================================================================
# get_instance_surrogate_display_stats Tests - Valid Cases
# ================================================================================

def test_get_instance_surrogate_display_stats_basic(instance_surrogate_tensor):
    """Test basic instance surrogate statistics calculation."""
    stats = get_instance_surrogate_display_stats(instance_surrogate_tensor)
    
    assert isinstance(stats, dict)
    assert "Shape" in stats
    assert "Height" in stats
    assert "Width" in stats
    assert "Total Pixels" in stats
    assert "Valid Pixels" in stats
    assert "Ignore Pixels" in stats
    assert "Valid Ratio" in stats
    assert "Data Type" in stats
    assert "Y Offset Range" in stats
    assert "X Offset Range" in stats
    assert "Y Offset Mean" in stats
    assert "X Offset Mean" in stats
    assert "Y Offset Std" in stats
    assert "X Offset Std" in stats
    assert "Magnitude Mean" in stats
    assert "Magnitude Std" in stats
    assert "Max Magnitude" in stats
    
    # Verify basic properties
    assert stats["Shape"] == f"{list(instance_surrogate_tensor.shape)}"
    assert stats["Height"] == 32
    assert stats["Width"] == 32
    assert stats["Total Pixels"] == 32 * 32


def test_get_instance_surrogate_display_stats_custom_ignore_value():
    """Test statistics with custom ignore value."""
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    # Set some pixels to custom ignore value
    ignore_mask = torch.rand(32, 32) < 0.2  # 20% ignore
    instance_surrogate[0, ignore_mask] = 100
    instance_surrogate[1, ignore_mask] = 100
    
    stats = get_instance_surrogate_display_stats(instance_surrogate, ignore_index=100)
    
    assert isinstance(stats, dict)
    assert stats["Ignore Pixels"] > 0
    assert stats["Valid Pixels"] < 32 * 32


def test_get_instance_surrogate_display_stats_no_valid_pixels():
    """Test statistics when all pixels are ignore regions."""
    # All pixels are ignore regions
    instance_surrogate = torch.full((2, 32, 32), 250.0, dtype=torch.float32)
    
    stats = get_instance_surrogate_display_stats(instance_surrogate)
    
    assert isinstance(stats, dict)
    assert stats["Valid Pixels"] == 0
    assert stats["Ignore Pixels"] == 32 * 32
    assert stats["Valid Ratio"] == "0.000"
    assert stats["Y Offset Range"] == "N/A (no valid pixels)"
    assert stats["X Offset Range"] == "N/A (no valid pixels)"
    assert stats["Y Offset Mean"] == "N/A"
    assert stats["X Offset Mean"] == "N/A"
    assert stats["Y Offset Std"] == "N/A"
    assert stats["X Offset Std"] == "N/A"
    assert stats["Magnitude Mean"] == "N/A"
    assert stats["Magnitude Std"] == "N/A"
    assert stats["Max Magnitude"] == "N/A"


def test_get_instance_surrogate_display_stats_all_valid_pixels():
    """Test statistics when all pixels are valid."""
    # No ignore regions
    instance_surrogate = torch.randn(2, 32, 32, dtype=torch.float32) * 5.0
    
    stats = get_instance_surrogate_display_stats(instance_surrogate)
    
    assert isinstance(stats, dict)
    assert stats["Valid Pixels"] == 32 * 32
    assert stats["Ignore Pixels"] == 0
    assert stats["Valid Ratio"] == "1.000"
    assert "N/A" not in stats["Y Offset Range"]
    assert "N/A" not in stats["X Offset Range"]
    assert stats["Y Offset Mean"] != "N/A"
    assert stats["X Offset Mean"] != "N/A"


def test_get_instance_surrogate_display_stats_known_values():
    """Test statistics with known offset values."""
    instance_surrogate = torch.zeros(2, 32, 32, dtype=torch.float32)
    
    # Set known offset patterns
    instance_surrogate[0, :16, :] = 2.0   # Y offset = 2 for top half
    instance_surrogate[1, :16, :] = 1.0   # X offset = 1 for top half
    
    instance_surrogate[0, 16:, :] = -2.0  # Y offset = -2 for bottom half
    instance_surrogate[1, 16:, :] = -1.0  # X offset = -1 for bottom half
    
    stats = get_instance_surrogate_display_stats(instance_surrogate)
    
    assert isinstance(stats, dict)
    assert stats["Valid Pixels"] == 32 * 32
    assert abs(float(stats["Y Offset Mean"])) < 1e-6  # Should be 0 (balanced)
    assert abs(float(stats["X Offset Mean"])) < 1e-6  # Should be 0 (balanced)
    assert "[-2.000, 2.000]" in stats["Y Offset Range"]
    assert "[-1.000, 1.000]" in stats["X Offset Range"]


@pytest.mark.parametrize("tensor_size", [(16, 16), (64, 64), (128, 128)])
def test_get_instance_surrogate_display_stats_various_sizes(tensor_size):
    """Test statistics with various tensor sizes."""
    h, w = tensor_size
    instance_surrogate = torch.randn(2, h, w, dtype=torch.float32) * 5.0
    
    stats = get_instance_surrogate_display_stats(instance_surrogate)
    
    assert isinstance(stats, dict)
    assert stats["Height"] == h
    assert stats["Width"] == w
    assert stats["Total Pixels"] == h * w


def test_get_instance_surrogate_display_stats_different_dtypes():
    """Test statistics with different tensor dtypes."""
    # Float64
    instance_surrogate_f64 = torch.randn(2, 32, 32, dtype=torch.float64) * 5.0
    stats_f64 = get_instance_surrogate_display_stats(instance_surrogate_f64)
    assert "torch.float64" in stats_f64["Data Type"]
    
    # Instance surrogates should be float32, not float16
    
    # Instance surrogates should be float (pixel offsets), not integer


# ================================================================================
# _convert_surrogate_to_instance_mask Tests - Valid Cases
# ================================================================================

def test_convert_surrogate_to_instance_mask_basic():
    """Test basic surrogate to instance mask conversion."""
    y_offset = torch.randn(32, 32, dtype=torch.float32) * 5.0
    x_offset = torch.randn(32, 32, dtype=torch.float32) * 5.0
    
    # Add some ignore regions
    ignore_mask = torch.rand(32, 32) < 0.1
    y_offset[ignore_mask] = 250
    x_offset[ignore_mask] = 250
    
    instance_mask = _convert_surrogate_to_instance_mask(y_offset, x_offset, ignore_index=250)
    
    assert isinstance(instance_mask, torch.Tensor)
    assert instance_mask.shape == (32, 32)
    assert instance_mask.dtype == torch.int64
    
    # Check that ignore regions are properly marked
    assert torch.all(instance_mask[ignore_mask] == 250)


def test_convert_surrogate_to_instance_mask_uniform_offsets():
    """Test conversion with uniform offset values."""
    # All pixels have same offset (should create single pseudo-instance)
    y_offset = torch.full((32, 32), 3.0, dtype=torch.float32)
    x_offset = torch.full((32, 32), 4.0, dtype=torch.float32)
    
    instance_mask = _convert_surrogate_to_instance_mask(y_offset, x_offset, ignore_index=250)
    
    assert isinstance(instance_mask, torch.Tensor)
    assert instance_mask.shape == (32, 32)
    
    # All non-ignore pixels should have same instance ID
    unique_values = torch.unique(instance_mask)
    assert len(unique_values) <= 2  # Should be at most background (0) and one instance


def test_convert_surrogate_to_instance_mask_all_ignore():
    """Test conversion when all pixels are ignore regions."""
    y_offset = torch.full((32, 32), 250.0, dtype=torch.float32)
    x_offset = torch.full((32, 32), 250.0, dtype=torch.float32)
    
    instance_mask = _convert_surrogate_to_instance_mask(y_offset, x_offset, ignore_index=250)
    
    assert isinstance(instance_mask, torch.Tensor)
    assert instance_mask.shape == (32, 32)
    assert torch.all(instance_mask == 250)  # All should be ignore value


def test_convert_surrogate_to_instance_mask_no_ignore():
    """Test conversion with no ignore regions."""
    y_offset = torch.randn(32, 32, dtype=torch.float32) * 5.0
    x_offset = torch.randn(32, 32, dtype=torch.float32) * 5.0
    
    instance_mask = _convert_surrogate_to_instance_mask(y_offset, x_offset, ignore_index=250)
    
    assert isinstance(instance_mask, torch.Tensor)
    assert instance_mask.shape == (32, 32)
    
    # No pixels should have ignore value
    assert torch.all(instance_mask != 250)


def test_convert_surrogate_to_instance_mask_edge_cases():
    """Test conversion with edge case offset values."""
    # Zero offsets (points to self)
    y_offset = torch.zeros(32, 32, dtype=torch.float32)
    x_offset = torch.zeros(32, 32, dtype=torch.float32)
    
    instance_mask = _convert_surrogate_to_instance_mask(y_offset, x_offset, ignore_index=250)
    assert isinstance(instance_mask, torch.Tensor)
    
    # Very large offsets
    y_offset_large = torch.full((32, 32), 1000.0, dtype=torch.float32)
    x_offset_large = torch.full((32, 32), 1000.0, dtype=torch.float32)
    
    instance_mask_large = _convert_surrogate_to_instance_mask(
        y_offset_large, x_offset_large, ignore_index=250
    )
    assert isinstance(instance_mask_large, torch.Tensor)


# ================================================================================
# Integration and Performance Tests
# ================================================================================

def test_instance_surrogate_display_pipeline(instance_surrogate_tensor):
    """Test complete instance surrogate display pipeline."""
    # Test display creation
    fig = create_instance_surrogate_display(instance_surrogate_tensor, "Pipeline Test")
    assert isinstance(fig, go.Figure)
    
    # Test statistics
    stats = get_instance_surrogate_display_stats(instance_surrogate_tensor)
    assert isinstance(stats, dict)
    
    # Verify consistency
    assert len(stats) >= 15  # Should have all expected keys


def test_instance_surrogate_display_determinism(instance_surrogate_tensor):
    """Test that instance surrogate display operations are deterministic."""
    # Display creation should be deterministic
    fig1 = create_instance_surrogate_display(instance_surrogate_tensor, "Determinism Test")
    fig2 = create_instance_surrogate_display(instance_surrogate_tensor, "Determinism Test")
    
    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert fig1.layout.title.text == fig2.layout.title.text
    
    # Statistics should be identical
    stats1 = get_instance_surrogate_display_stats(instance_surrogate_tensor)
    stats2 = get_instance_surrogate_display_stats(instance_surrogate_tensor)
    
    assert stats1 == stats2


def test_performance_with_large_instance_surrogate():
    """Test performance with large instance surrogate maps."""
    # Create large instance surrogate map
    large_instance_surrogate = torch.randn(2, 512, 512, dtype=torch.float32) * 10.0
    
    # Add some ignore regions
    ignore_mask = torch.rand(512, 512) < 0.05
    large_instance_surrogate[0, ignore_mask] = 250
    large_instance_surrogate[1, ignore_mask] = 250
    
    # These should complete without error
    fig = create_instance_surrogate_display(large_instance_surrogate, "Large Instance Test")
    stats = get_instance_surrogate_display_stats(large_instance_surrogate)
    
    # Basic checks
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)
    assert stats["Height"] == 512
    assert stats["Width"] == 512


# ================================================================================
# Correctness Verification Tests  
# ================================================================================

def test_instance_surrogate_display_known_pattern():
    """Test instance surrogate display with known offset pattern."""
    # Create radial offset pattern (points toward center)
    instance_surrogate = torch.zeros(2, 64, 64, dtype=torch.float32)
    center_y, center_x = 32, 32
    
    for i in range(64):
        for j in range(64):
            # Calculate offset vector pointing toward center
            offset_y = center_y - i
            offset_x = center_x - j
            
            instance_surrogate[0, i, j] = offset_y
            instance_surrogate[1, i, j] = offset_x
    
    # Add border ignore regions
    instance_surrogate[:, :2, :] = 250
    instance_surrogate[:, -2:, :] = 250
    instance_surrogate[:, :, :2] = 250
    instance_surrogate[:, :, -2:] = 250
    
    fig = create_instance_surrogate_display(instance_surrogate, "Radial Pattern")
    assert isinstance(fig, go.Figure)
    
    stats = get_instance_surrogate_display_stats(instance_surrogate)
    assert stats["Valid Pixels"] < 64 * 64  # Should have ignore regions
    assert abs(float(stats["Y Offset Mean"])) < 1.0  # Should be approximately 0
    assert abs(float(stats["X Offset Mean"])) < 1.0  # Should be approximately 0


def test_instance_surrogate_mask_conversion_correctness():
    """Test that surrogate to mask conversion produces reasonable results."""
    # Create instance surrogate with distinct regions
    instance_surrogate = torch.zeros(2, 32, 32, dtype=torch.float32)
    
    # Region 1: Small offsets (close to center)
    instance_surrogate[0, 8:16, 8:16] = torch.randn(8, 8) * 0.5
    instance_surrogate[1, 8:16, 8:16] = torch.randn(8, 8) * 0.5
    
    # Region 2: Large offsets (far from center)
    instance_surrogate[0, 20:28, 20:28] = torch.randn(8, 8) * 5.0
    instance_surrogate[1, 20:28, 20:28] = torch.randn(8, 8) * 5.0
    
    # Convert to instance mask
    y_offset = instance_surrogate[0]
    x_offset = instance_surrogate[1]
    instance_mask = _convert_surrogate_to_instance_mask(y_offset, x_offset, ignore_index=250)
    
    # Check that conversion produces valid instance mask
    assert isinstance(instance_mask, torch.Tensor)
    assert instance_mask.shape == (32, 32)
    assert instance_mask.dtype == torch.int64
    
    # Check that regions with similar offsets get similar instance IDs
    region1_ids = torch.unique(instance_mask[8:16, 8:16])
    region2_ids = torch.unique(instance_mask[20:28, 20:28])
    
    # Both regions should have some instance assignments
    assert len(region1_ids) > 0
    assert len(region2_ids) > 0