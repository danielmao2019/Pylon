"""Tests for normal display functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.normal_display import (
    create_normal_display,
    get_normal_display_stats
)


# ================================================================================
# create_normal_display Tests - Valid Cases
# ================================================================================

def test_create_normal_display_basic(normal_tensor):
    """Test basic normal display creation."""
    fig = create_normal_display(normal_tensor, "Test Normal Display")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Normal Display"
    assert fig.layout.height == 400


def test_create_normal_display_various_sizes():
    """Test normal display with various tensor sizes."""
    sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]

    for h, w in sizes:
        normals = torch.randn(3, h, w, dtype=torch.float32)
        # Normalize to unit vectors
        magnitude = torch.sqrt((normals ** 2).sum(dim=0, keepdim=True))
        magnitude = torch.clamp(magnitude, min=1e-8)
        normals = normals / magnitude

        fig = create_normal_display(normals, f"Test {h}x{w}")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == f"Test {h}x{w}"


def test_create_normal_display_extreme_normal_values():
    """Test normal display with extreme normal vector values."""
    # Very large magnitude normals (will be normalized)
    large_normals = torch.full((3, 32, 32), 1000.0, dtype=torch.float32)
    fig = create_normal_display(large_normals, "Large Normals")
    assert isinstance(fig, go.Figure)

    # Very small magnitude normals
    small_normals = torch.full((3, 32, 32), 1e-6, dtype=torch.float32)
    fig = create_normal_display(small_normals, "Small Normals")
    assert isinstance(fig, go.Figure)

    # Mixed positive and negative
    mixed_normals = torch.randn(3, 32, 32, dtype=torch.float32) * 100
    fig = create_normal_display(mixed_normals, "Mixed Normals")
    assert isinstance(fig, go.Figure)


def test_create_normal_display_unit_normals():
    """Test normal display with perfect unit normal vectors."""
    # Create unit normals pointing in different directions
    normals = torch.zeros(3, 32, 32, dtype=torch.float32)

    # X-direction normals
    normals[0, :16, :] = 1.0

    # Y-direction normals
    normals[1, 16:, :16] = 1.0

    # Z-direction normals
    normals[2, 16:, 16:] = 1.0

    fig = create_normal_display(normals, "Unit Normals")
    assert isinstance(fig, go.Figure)


def test_create_normal_display_with_kwargs(normal_tensor):
    """Test normal display with additional keyword arguments."""
    fig = create_normal_display(
        normal_tensor,
        "Test with Kwargs",
        extra_param="ignored"  # Should be ignored
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test with Kwargs"


# ================================================================================
# Integration and Performance Tests
# ================================================================================

def test_normal_display_pipeline(normal_tensor):
    """Test complete normal display pipeline."""
    # Test display creation
    fig = create_normal_display(normal_tensor, "Pipeline Test")
    assert isinstance(fig, go.Figure)

    # Test statistics
    stats = get_normal_display_stats(normal_tensor)
    assert isinstance(stats, dict)

    # Verify consistency
    assert len(stats) >= 9  # Should have all expected keys


def test_normal_display_determinism(normal_tensor):
    """Test that normal display operations are deterministic."""
    # Display creation should be deterministic
    fig1 = create_normal_display(normal_tensor, "Determinism Test")
    fig2 = create_normal_display(normal_tensor, "Determinism Test")

    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert fig1.layout.title.text == fig2.layout.title.text

    # Statistics should be identical
    stats1 = get_normal_display_stats(normal_tensor)
    stats2 = get_normal_display_stats(normal_tensor)

    assert stats1 == stats2


def test_performance_with_large_normals():
    """Test performance with large normal maps."""
    # Create large normal map
    large_normals = torch.randn(3, 512, 512, dtype=torch.float32)

    # These should complete without error
    fig = create_normal_display(large_normals, "Large Normal Test")
    stats = get_normal_display_stats(large_normals)

    # Basic checks
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)
    assert stats["shape"] == [3, 512, 512]


# ================================================================================
# Correctness Verification Tests
# ================================================================================

def test_normal_display_rgb_mapping_correctness():
    """Test that normal-to-RGB mapping works correctly."""
    # Create known normal vectors and verify they map correctly
    normals = torch.zeros(3, 2, 2, dtype=torch.float32)

    # Pure X normal (+1, 0, 0) should map to specific RGB
    normals[0, 0, 0] = 1.0
    normals[1, 0, 0] = 0.0
    normals[2, 0, 0] = 0.0

    # Pure Y normal (0, +1, 0) should map to specific RGB
    normals[0, 0, 1] = 0.0
    normals[1, 0, 1] = 1.0
    normals[2, 0, 1] = 0.0

    # Pure Z normal (0, 0, +1) should map to specific RGB
    normals[0, 1, 0] = 0.0
    normals[1, 1, 0] = 0.0
    normals[2, 1, 0] = 1.0

    # Negative Z normal (0, 0, -1) should map differently
    normals[0, 1, 1] = 0.0
    normals[1, 1, 1] = 0.0
    normals[2, 1, 1] = -1.0

    fig = create_normal_display(normals, "RGB Mapping Test")
    assert isinstance(fig, go.Figure)

    stats = get_normal_display_stats(normals)
    assert stats["valid_pixels"] == 4
    assert abs(stats["mean_magnitude"] - 1.0) < 1e-6  # All unit vectors


def test_normal_display_range_handling():
    """Test that normal display handles [-1,1] to [0,1] range conversion correctly."""
    # Create normals that span the full [-1, 1] range
    normals = torch.zeros(3, 3, 3, dtype=torch.float32)

    # Extreme negative values
    normals[:, 0, 0] = -1.0

    # Zero values
    normals[:, 1, 1] = 0.0

    # Extreme positive values
    normals[:, 2, 2] = 1.0

    fig = create_normal_display(normals, "Range Test")
    assert isinstance(fig, go.Figure)

    stats = get_normal_display_stats(normals)
    assert isinstance(stats, dict)
    assert stats["valid_pixels"] == 2  # Only non-zero normals are valid for statistics