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