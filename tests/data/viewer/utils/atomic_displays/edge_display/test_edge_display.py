"""Tests for edge display functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any

import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.edge_display import (
    create_edge_display,
    get_edge_display_stats
)


# ================================================================================
# create_edge_display Tests - Valid Cases
# ================================================================================

def test_create_edge_display_2d_tensor(edge_tensor_2d):
    """Test edge display creation with 2D tensor [H, W].""" 
    fig = create_edge_display(edge_tensor_2d, "Test Edge Display 2D")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Edge Display 2D"
    assert fig.layout.height == 400


def test_create_edge_display_2d_tensor(edge_tensor_2d):
    """Test edge display creation with 2D tensor [H, W]."""
    fig = create_edge_display(edge_tensor_2d, "Test Edge Display 2D")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Edge Display 2D"
    assert fig.layout.height == 400


@pytest.mark.parametrize("colorscale", ["greys", "viridis", "plasma", "hot", "turbo"])
def test_create_edge_display_various_colorscales(edge_tensor, colorscale):
    """Test edge display with various colorscales."""
    fig = create_edge_display(edge_tensor, "Test Colorscales", colorscale=colorscale)
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Colorscales"


def test_create_edge_display_binary_edges():
    """Test edge display with binary edge values (0 and 1)."""
    # Create binary edge pattern
    edges = torch.zeros(32, 32, dtype=torch.float32)
    edges[::4, :] = 1.0  # Horizontal lines every 4 pixels
    edges[:, ::4] = 1.0  # Vertical lines every 4 pixels
    
    fig = create_edge_display(edges, "Binary Edges")
    assert isinstance(fig, go.Figure)


def test_create_edge_display_continuous_edges():
    """Test edge display with continuous edge strength values."""
    # Create continuous edge strength values [0, 1]
    edges = torch.rand(32, 32, dtype=torch.float32)
    
    fig = create_edge_display(edges, "Continuous Edges")
    assert isinstance(fig, go.Figure)


def test_create_edge_display_with_kwargs(edge_tensor):
    """Test edge display with additional keyword arguments."""
    fig = create_edge_display(
        edge_tensor,
        "Test with Kwargs",
        colorscale="Gray",
        extra_param="ignored"  # Should be ignored
    )
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test with Kwargs"


@pytest.mark.parametrize("tensor_size", [(16, 16), (64, 64), (128, 128)])
def test_create_edge_display_various_sizes(tensor_size):
    """Test edge display with various tensor sizes."""
    h, w = tensor_size
    edges = torch.rand(h, w, dtype=torch.float32)
    
    fig = create_edge_display(edges, f"Test {h}x{w}")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == f"Test {h}x{w}"


def test_create_edge_display_extreme_values():
    """Test edge display with extreme edge values."""
    # Very large values
    large_edges = torch.full((32, 32), 1000.0, dtype=torch.float32)
    fig = create_edge_display(large_edges, "Large Values")
    assert isinstance(fig, go.Figure)
    
    # Very small values
    small_edges = torch.full((32, 32), 1e-6, dtype=torch.float32)
    fig = create_edge_display(small_edges, "Small Values")
    assert isinstance(fig, go.Figure)
    
    # Negative values
    negative_edges = torch.full((32, 32), -0.5, dtype=torch.float32)
    fig = create_edge_display(negative_edges, "Negative Values")
    assert isinstance(fig, go.Figure)


# ================================================================================
# Note: Stats tests moved to dedicated test_edge_display_stats.py file


# Integration and Performance Tests
# ================================================================================

def test_edge_display_pipeline(edge_tensor):
    """Test complete edge display pipeline."""
    # Test display creation
    fig = create_edge_display(edge_tensor, "Pipeline Test")
    assert isinstance(fig, go.Figure)
    
    # Test statistics
    stats = get_edge_display_stats(edge_tensor)
    assert isinstance(stats, dict)
    
    # Verify consistency
    assert len(stats) >= 9  # Should have all expected keys


def test_edge_display_determinism(edge_tensor):
    """Test that edge display operations are deterministic."""
    # Display creation should be deterministic
    fig1 = create_edge_display(edge_tensor, "Determinism Test")
    fig2 = create_edge_display(edge_tensor, "Determinism Test")
    
    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert fig1.layout.title.text == fig2.layout.title.text
    
    # Statistics should be identical
    stats1 = get_edge_display_stats(edge_tensor)
    stats2 = get_edge_display_stats(edge_tensor)
    
    assert stats1 == stats2


def test_performance_with_large_edges():
    """Test performance with large edge maps."""
    # Create large edge map
    large_edges = torch.rand(512, 512, dtype=torch.float32)
    
    # These should complete without error
    fig = create_edge_display(large_edges, "Large Edge Test")
    stats = get_edge_display_stats(large_edges)
    
    # Basic checks
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)
    assert stats["shape"] == [512, 512]


# ================================================================================
# Correctness Verification Tests
# ================================================================================

def test_edge_display_known_patterns():
    """Test edge display with known edge patterns."""
    # Create checkerboard edge pattern
    edges = torch.zeros(32, 32, dtype=torch.float32)
    edges[::2, ::2] = 1.0  # Even rows, even cols
    edges[1::2, 1::2] = 1.0  # Odd rows, odd cols
    
    fig = create_edge_display(edges, "Checkerboard")
    assert isinstance(fig, go.Figure)
    
    stats = get_edge_display_stats(edges)
    assert abs(stats["edge_percentage"] - 50.0) < 0.1  # Approximately 50% should be edges


def test_edge_display_gradient_pattern():
    """Test edge display with gradient edge strengths."""
    # Create gradient from 0 to 1
    edges = torch.zeros(32, 32, dtype=torch.float32)
    for i in range(32):
        edges[i, :] = i / 31.0  # Linear gradient
    
    fig = create_edge_display(edges, "Gradient")
    assert isinstance(fig, go.Figure)
    
    stats = get_edge_display_stats(edges)
    assert abs(stats["min_edge"] - 0.0) < 1e-6
    assert abs(stats["max_edge"] - 1.0) < 1e-6
    assert 0.4 < stats["mean_edge"] < 0.6  # Should be approximately 0.5


def test_edge_display_3d_to_2d_conversion():
    """Test that 3D tensors are correctly converted to 2D for display."""
    # Create 3D tensor with single channel
    edges_3d = torch.rand(1, 32, 32, dtype=torch.float32)
    
    fig = create_edge_display(edges_3d, "3D to 2D Test")
    assert isinstance(fig, go.Figure)
    
    stats = get_edge_display_stats(edges_3d)
    assert len(stats["shape"]) == 2  # Should be squeezed to 2D
    assert stats["shape"] == [32, 32]
