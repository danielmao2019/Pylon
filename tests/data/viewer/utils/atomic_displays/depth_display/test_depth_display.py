"""Tests for depth display functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.depth_display import (
    create_depth_display,
    get_depth_display_stats
)


# ================================================================================
# create_depth_display Tests - Valid Cases
# ================================================================================

def test_create_depth_display_basic(depth_tensor):
    """Test basic depth display creation."""
    fig = create_depth_display(depth_tensor, "Test Depth Display")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Depth Display"
    assert fig.layout.height == 400


@pytest.mark.parametrize("colorscale", ["Viridis", "Plasma", "Inferno", "Cividis", "Turbo"])
def test_create_depth_display_various_colorscales(depth_tensor, colorscale):
    """Test depth display with various colorscales."""
    fig = create_depth_display(depth_tensor, "Test Colorscales", colorscale=colorscale)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Colorscales"


def test_create_depth_display_realistic_depth_values():
    """Test depth display with realistic depth values (meters)."""
    # Depth values in meters (0.1m to 100m)
    depth = torch.rand(64, 64, dtype=torch.float32) * 99.9 + 0.1

    fig = create_depth_display(depth, "Realistic Depths")
    assert isinstance(fig, go.Figure)


def test_create_depth_display_close_range():
    """Test depth display with close-range depth values."""
    # Very close objects (1cm to 1m)
    depth = torch.rand(32, 32, dtype=torch.float32) * 0.99 + 0.01

    fig = create_depth_display(depth, "Close Range")
    assert isinstance(fig, go.Figure)


def test_create_depth_display_far_range():
    """Test depth display with far-range depth values."""
    # Very far objects (100m to 1000m)
    depth = torch.rand(32, 32, dtype=torch.float32) * 900.0 + 100.0

    fig = create_depth_display(depth, "Far Range")
    assert isinstance(fig, go.Figure)


def test_create_depth_display_with_kwargs(depth_tensor):
    """Test depth display with additional keyword arguments."""
    fig = create_depth_display(
        depth_tensor,
        "Test with Kwargs",
        colorscale="Viridis",
        extra_param="ignored"  # Should be ignored
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test with Kwargs"


@pytest.mark.parametrize("tensor_size", [(16, 16), (64, 64), (128, 128), (256, 256)])
def test_create_depth_display_various_sizes(tensor_size):
    """Test depth display with various tensor sizes."""
    h, w = tensor_size
    depth = torch.rand(h, w, dtype=torch.float32) * 10.0 + 0.1

    fig = create_depth_display(depth, f"Test {h}x{w}")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == f"Test {h}x{w}"


def test_create_depth_display_uniform_depth():
    """Test depth display with uniform depth values."""
    # All pixels at same depth
    depth = torch.full((32, 32), 5.0, dtype=torch.float32)

    fig = create_depth_display(depth, "Uniform Depth")
    assert isinstance(fig, go.Figure)


def test_create_depth_display_depth_gradient():
    """Test depth display with depth gradient."""
    # Create depth gradient from near to far
    depth = torch.zeros(32, 32, dtype=torch.float32)
    for i in range(32):
        depth[i, :] = 0.1 + (i / 31.0) * 9.9  # Linear gradient 0.1 to 10.0

    fig = create_depth_display(depth, "Depth Gradient")
    assert isinstance(fig, go.Figure)


# ================================================================================
# Integration and Performance Tests
# ================================================================================

def test_depth_display_pipeline(depth_tensor):
    """Test basic depth statistics calculation."""
    stats = get_depth_display_stats(depth_tensor)

    assert isinstance(stats, dict)
    assert "shape" in stats
    assert "dtype" in stats
    assert "valid_pixels" in stats
    assert "total_pixels" in stats
    assert "min_depth" in stats
    assert "max_depth" in stats
    assert "mean_depth" in stats
    assert "std_depth" in stats

    # Verify basic properties
    assert stats["shape"] == [32, 32]
    assert stats["total_pixels"] == 32 * 32
    assert isinstance(stats["valid_pixels"], int)
    assert stats["valid_pixels"] <= stats["total_pixels"]


def test_get_depth_display_stats_realistic_depths():
    """Test statistics with realistic depth values."""
    # Depth values representing a typical indoor scene (0.5m to 5m)
    depth = torch.rand(64, 64, dtype=torch.float32) * 4.5 + 0.5

    stats = get_depth_display_stats(depth)

    assert isinstance(stats, dict)
    assert stats["valid_pixels"] == 64 * 64  # All should be valid
    assert 0.5 <= stats["min_depth"] <= 1.0  # Should be close to 0.5
    assert 4.5 <= stats["max_depth"] <= 5.0  # Should be close to 5.0
    assert 2.0 <= stats["mean_depth"] <= 3.0  # Should be around 2.75


def test_get_depth_display_stats_with_invalid_depths():
    """Test statistics with invalid depth values."""
    depth = torch.rand(32, 32, dtype=torch.float32) * 10.0 + 0.1

    # Add some zero/negative depths (invalid)
    depth[:5, :5] = 0.0
    depth[5:10, 5:10] = -1.0

    # Add some NaN values
    depth[10:15, 10:15] = float('nan')

    # Add some infinity values
    depth[15:20, 15:20] = float('inf')

    stats = get_depth_display_stats(depth)

    assert isinstance(stats, dict)
    assert stats["valid_pixels"] < 32 * 32  # Some pixels should be invalid
    assert stats["valid_pixels"] >= 0


def test_get_depth_display_stats_all_invalid():
    """Test statistics when all depth values are invalid."""
    # All NaN depths
    depth = torch.full((32, 32), float('nan'), dtype=torch.float32)

    stats = get_depth_display_stats(depth)

    assert isinstance(stats, dict)
    assert stats["valid_pixels"] == 0
    assert stats["total_pixels"] == 32 * 32
    assert stats["min_depth"] == "N/A"
    assert stats["max_depth"] == "N/A"
    assert stats["mean_depth"] == "N/A"
    assert stats["std_depth"] == "N/A"


def test_get_depth_display_stats_zero_negative_depths():
    """Test statistics filtering out zero and negative depths."""
    depth = torch.ones(32, 32, dtype=torch.float32) * 5.0

    # Set some pixels to zero and negative values
    depth[:10, :10] = 0.0  # Zero depths (invalid)
    depth[10:20, 10:20] = -2.0  # Negative depths (invalid)

    stats = get_depth_display_stats(depth)

    assert isinstance(stats, dict)
    assert stats["valid_pixels"] == 32 * 32 - 200  # Should exclude 200 invalid pixels
    assert stats["min_depth"] == 5.0
    assert stats["max_depth"] == 5.0
    assert stats["mean_depth"] == 5.0


@pytest.mark.parametrize("tensor_size", [(16, 16), (64, 64), (128, 128)])
def test_get_depth_display_stats_various_sizes(tensor_size):
    """Test statistics with various tensor sizes."""
    h, w = tensor_size
    depth = torch.rand(h, w, dtype=torch.float32) * 10.0 + 0.1

    stats = get_depth_display_stats(depth)

    assert isinstance(stats, dict)
    assert stats["shape"] == [h, w]
    assert stats["total_pixels"] == h * w


def test_get_depth_display_stats_single_pixel():
    """Test statistics with single pixel depth."""
    depth = torch.tensor([[2.5]], dtype=torch.float32)

    stats = get_depth_display_stats(depth)

    assert isinstance(stats, dict)
    assert stats["shape"] == [1, 1]
    assert stats["valid_pixels"] == 1
    assert stats["total_pixels"] == 1
    assert stats["min_depth"] == 2.5
    assert stats["max_depth"] == 2.5
    assert stats["mean_depth"] == 2.5
    import math
    assert math.isnan(stats["std_depth"])  # Single pixel has NaN std


def test_get_depth_display_stats_different_dtypes():
    """Test statistics with different tensor dtypes."""
    # Float32 (default)
    depth_f32 = torch.rand(32, 32, dtype=torch.float32) * 10.0 + 0.1
    stats_f32 = get_depth_display_stats(depth_f32)
    assert "torch.float32" in stats_f32["dtype"]

    # Float64 (acceptable for high-precision depth)
    depth_f64 = torch.rand(32, 32, dtype=torch.float64) * 10.0 + 0.1
    stats_f64 = get_depth_display_stats(depth_f64)
    assert "torch.float64" in stats_f64["dtype"]


# ================================================================================
# Integration and Performance Tests
# ================================================================================

def test_depth_display_pipeline(depth_tensor):
    """Test complete depth display pipeline."""
    # Test display creation
    fig = create_depth_display(depth_tensor, "Pipeline Test")
    assert isinstance(fig, go.Figure)

    # Test statistics
    stats = get_depth_display_stats(depth_tensor)
    assert isinstance(stats, dict)

    # Verify consistency
    assert len(stats) >= 8  # Should have all expected keys


def test_depth_display_determinism(depth_tensor):
    """Test that depth display operations are deterministic."""
    # Display creation should be deterministic
    fig1 = create_depth_display(depth_tensor, "Determinism Test")
    fig2 = create_depth_display(depth_tensor, "Determinism Test")

    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert fig1.layout.title.text == fig2.layout.title.text

    # Statistics should be identical
    stats1 = get_depth_display_stats(depth_tensor)
    stats2 = get_depth_display_stats(depth_tensor)

    assert stats1 == stats2


def test_performance_with_large_depth_maps():
    """Test performance with large depth maps."""
    # Create large depth map
    large_depth = torch.rand(512, 512, dtype=torch.float32) * 10.0 + 0.1

    # These should complete without error
    fig = create_depth_display(large_depth, "Large Depth Test")
    stats = get_depth_display_stats(large_depth)

    # Basic checks
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)
    assert stats["shape"] == [512, 512]


# ================================================================================
# Correctness Verification Tests
# ================================================================================

def test_depth_display_known_depth_pattern():
    """Test depth display with known depth pattern."""
    # Create concentric depth rings (like a bowl)
    depth = torch.zeros(64, 64, dtype=torch.float32)
    center_y, center_x = 32, 32

    for i in range(64):
        for j in range(64):
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            depth[i, j] = 1.0 + distance * 0.1  # Depth increases with distance from center

    fig = create_depth_display(depth, "Concentric Depth")
    assert isinstance(fig, go.Figure)

    stats = get_depth_display_stats(depth)
    assert stats["valid_pixels"] == 64 * 64  # All pixels valid
    assert stats["min_depth"] >= 1.0  # Center depth
    assert stats["max_depth"] <= 10.0  # Max corner depth


def test_depth_display_step_function():
    """Test depth display with step function depths."""
    depth = torch.zeros(32, 32, dtype=torch.float32)

    # Create depth steps
    depth[:8, :] = 1.0    # Near objects
    depth[8:16, :] = 3.0   # Medium distance
    depth[16:24, :] = 5.0  # Far objects
    depth[24:, :] = 10.0   # Very far objects

    fig = create_depth_display(depth, "Step Depths")
    assert isinstance(fig, go.Figure)

    stats = get_depth_display_stats(depth)
    assert stats["min_depth"] == 1.0
    assert stats["max_depth"] == 10.0
    assert stats["mean_depth"] == 4.75  # (1+3+5+10)/4


def test_depth_display_extreme_depth_ranges():
    """Test depth display with extreme but valid depth ranges."""
    # Very close depths (millimeter scale)
    close_depth = torch.rand(32, 32, dtype=torch.float32) * 0.001 + 0.001
    fig = create_depth_display(close_depth, "Millimeter Depths")
    assert isinstance(fig, go.Figure)

    stats = get_depth_display_stats(close_depth)
    assert stats["min_depth"] >= 0.001
    assert stats["max_depth"] <= 0.002

    # Very far depths (kilometer scale)
    far_depth = torch.rand(32, 32, dtype=torch.float32) * 1000.0 + 1000.0
    fig = create_depth_display(far_depth, "Kilometer Depths")
    assert isinstance(fig, go.Figure)

    stats = get_depth_display_stats(far_depth)
    assert stats["min_depth"] >= 1000.0
    assert stats["max_depth"] <= 2000.0


def test_depth_display_invalid_filtering():
    """Test that depth display correctly filters invalid depths."""
    # Mix of valid and invalid depths
    depth = torch.ones(20, 20, dtype=torch.float32) * 5.0

    # Add various invalid values
    depth[0:5, 0:5] = 0.0      # Zero depths
    depth[5:10, 0:5] = -1.0    # Negative depths
    depth[0:5, 5:10] = float('nan')  # NaN depths
    depth[5:10, 5:10] = float('inf') # Infinite depths

    stats = get_depth_display_stats(depth)

    # Only pixels with depth > 0 and finite should be valid
    expected_valid = 20 * 20 - 100  # Total minus 4 invalid 5x5 regions
    assert stats["valid_pixels"] == expected_valid
    assert stats["min_depth"] == 5.0
    assert stats["max_depth"] == 5.0