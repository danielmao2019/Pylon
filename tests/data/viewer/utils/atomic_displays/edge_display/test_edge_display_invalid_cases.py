"""Tests for edge display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.edge_display import (
    create_edge_display,
    get_edge_display_stats
)


# ================================================================================
# create_edge_display Tests - Invalid Cases
# ================================================================================

def test_create_edge_display_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display("not_a_tensor", "Test")

    assert "Expected torch.Tensor" in str(exc_info.value)


def test_create_edge_display_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    edges_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_1d, "Test")
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)

    # 4D tensor
    edges_4d = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_4d, "Test")
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)

    # 5D tensor
    edges_5d = torch.rand(1, 1, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_5d, "Test")
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)


def test_create_edge_display_invalid_batch_size():
    """Test assertion failure for 3D tensor with batch size > 1."""
    # Batch size 2
    edges_batch2 = torch.rand(2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_batch2, "Test")
    assert "Expected batch size 1 for visualization, got 2" in str(exc_info.value)

    # Batch size 3
    edges_batch3 = torch.rand(3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_batch3, "Test")
    assert "Expected batch size 1 for visualization, got 3" in str(exc_info.value)

    # Batch size 4
    edges_batch4 = torch.rand(4, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_batch4, "Test")
    assert "Expected batch size 1 for visualization, got 4" in str(exc_info.value)


def test_create_edge_display_empty_tensor():
    """Test assertion failure for empty tensor."""
    # 2D empty tensor
    empty_edges_2d = torch.empty((0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(empty_edges_2d, "Test")
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # 3D empty tensor
    empty_edges_3d = torch.empty((1, 0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(empty_edges_3d, "Test")
    assert "Edge tensor cannot be empty" in str(exc_info.value)


def test_create_edge_display_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height (2D)
    edges_zero_h_2d = torch.empty((0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_zero_h_2d, "Test")
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # Zero width (2D)
    edges_zero_w_2d = torch.empty((32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_zero_w_2d, "Test")
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # Zero height (3D)
    edges_zero_h_3d = torch.empty((1, 0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_zero_h_3d, "Test")
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # Zero width (3D)
    edges_zero_w_3d = torch.empty((1, 32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges_zero_w_3d, "Test")
    assert "Edge tensor cannot be empty" in str(exc_info.value)


def test_create_edge_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    edges = torch.rand(32, 32, dtype=torch.float32)

    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges, 123)

    assert "Expected str title" in str(exc_info.value)


def test_create_edge_display_invalid_colorscale_type():
    """Test assertion failure for invalid colorscale type."""
    edges = torch.rand(32, 32, dtype=torch.float32)

    with pytest.raises(AssertionError) as exc_info:
        create_edge_display(edges, "Test", colorscale=123)

    assert "Expected str colorscale" in str(exc_info.value)


# ================================================================================
# get_edge_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_edge_display_stats_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats("not_a_tensor")

    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_edge_display_stats_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    # 1D tensor
    edges_1d = torch.rand(100, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_1d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)

    # 4D tensor
    edges_4d = torch.rand(1, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_4d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)

    # 5D tensor
    edges_5d = torch.rand(1, 1, 1, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_5d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)


def test_get_edge_display_stats_invalid_batch_size():
    """Test assertion failure for 3D tensor with batch size > 1."""
    # Batch size 2
    edges_batch2 = torch.rand(2, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_batch2)
    assert "Expected batch size 1 for analysis, got 2" in str(exc_info.value)

    # Batch size 3
    edges_batch3 = torch.rand(3, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_batch3)
    assert "Expected batch size 1 for analysis, got 3" in str(exc_info.value)

    # Batch size 4
    edges_batch4 = torch.rand(4, 32, 32, dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_batch4)
    assert "Expected batch size 1 for analysis, got 4" in str(exc_info.value)


def test_get_edge_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    # 2D empty tensor
    empty_edges_2d = torch.empty((0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(empty_edges_2d)
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # 3D empty tensor
    empty_edges_3d = torch.empty((1, 0, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(empty_edges_3d)
    assert "Edge tensor cannot be empty" in str(exc_info.value)


def test_get_edge_display_stats_zero_dimensions():
    """Test assertion failure for tensors with zero dimensions."""
    # Zero height (2D)
    edges_zero_h_2d = torch.empty((0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_zero_h_2d)
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # Zero width (2D)
    edges_zero_w_2d = torch.empty((32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_zero_w_2d)
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # Zero height (3D)
    edges_zero_h_3d = torch.empty((1, 0, 32), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_zero_h_3d)
    assert "Edge tensor cannot be empty" in str(exc_info.value)

    # Zero width (3D)
    edges_zero_w_3d = torch.empty((1, 32, 0), dtype=torch.float32)
    with pytest.raises(AssertionError) as exc_info:
        get_edge_display_stats(edges_zero_w_3d)
    assert "Edge tensor cannot be empty" in str(exc_info.value)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_create_edge_display_with_different_dtypes():
    """Test edge display with various tensor dtypes (should work)."""
    # Float64
    edges_f64 = torch.rand(32, 32, dtype=torch.float64)
    fig = create_edge_display(edges_f64, "Float64 Test")
    assert isinstance(fig, go.Figure)

    # Float16
    edges_f16 = torch.rand(32, 32, dtype=torch.float16)
    fig = create_edge_display(edges_f16, "Float16 Test")
    assert isinstance(fig, go.Figure)

    # Integer types (binary or multi-class edge labels)
    edges_int32 = torch.randint(0, 2, (32, 32), dtype=torch.int32)
    fig = create_edge_display(edges_int32, "Int32 Test")
    assert isinstance(fig, go.Figure)

    edges_int64 = torch.randint(0, 2, (32, 32), dtype=torch.int64)
    fig = create_edge_display(edges_int64, "Int64 Test")
    assert isinstance(fig, go.Figure)


def test_get_edge_display_stats_with_different_dtypes():
    """Test statistics with various tensor dtypes."""
    # Float64 (high precision edge detection)
    edges_f64 = torch.rand(32, 32, dtype=torch.float64)
    stats = get_edge_display_stats(edges_f64)
    assert isinstance(stats, dict)
    assert "torch.float64" in stats["dtype"]

    # Integer (binary or multi-class edge labels)
    edges_int = torch.randint(0, 2, (32, 32), dtype=torch.int32)
    stats = get_edge_display_stats(edges_int)
    assert isinstance(stats, dict)
    assert "torch.int32" in stats["dtype"]


def test_edge_display_with_extreme_tensor_shapes():
    """Test with extreme but valid tensor shapes."""
    # Very thin tensor (2D)
    thin_edges_2d = torch.rand(1, 1000, dtype=torch.float32)
    fig = create_edge_display(thin_edges_2d, "Thin 2D")
    assert isinstance(fig, go.Figure)

    # Very thin tensor (3D)
    thin_edges_3d = torch.rand(1, 1, 1000, dtype=torch.float32)
    fig = create_edge_display(thin_edges_3d, "Thin 3D")
    assert isinstance(fig, go.Figure)

    # Very tall tensor (2D)
    tall_edges_2d = torch.rand(1000, 1, dtype=torch.float32)
    fig = create_edge_display(tall_edges_2d, "Tall 2D")
    assert isinstance(fig, go.Figure)

    # Very tall tensor (3D)
    tall_edges_3d = torch.rand(1, 1000, 1, dtype=torch.float32)
    fig = create_edge_display(tall_edges_3d, "Tall 3D")
    assert isinstance(fig, go.Figure)


def test_edge_display_with_unusual_values():
    """Test edge display with unusual but valid values."""
    # All zeros (no edges)
    zero_edges = torch.zeros(32, 32, dtype=torch.float32)
    fig = create_edge_display(zero_edges, "No Edges")
    assert isinstance(fig, go.Figure)

    stats = get_edge_display_stats(zero_edges)
    assert stats["min_edge"] == 0.0
    assert stats["max_edge"] == 0.0
    assert stats["edge_percentage"] == 0.0

    # All ones (all edges)
    one_edges = torch.ones(32, 32, dtype=torch.float32)
    fig = create_edge_display(one_edges, "All Edges")
    assert isinstance(fig, go.Figure)

    stats = get_edge_display_stats(one_edges)
    assert stats["min_edge"] == 1.0
    assert stats["max_edge"] == 1.0


def test_edge_display_with_mixed_valid_invalid_values():
    """Test edge display with mix of valid and invalid values."""
    edges = torch.rand(32, 32, dtype=torch.float32)

    # Set some regions to NaN
    edges[:10, :10] = float('nan')

    # Set some regions to infinity
    edges[10:20, 10:20] = float('inf')

    # This should still work
    fig = create_edge_display(edges, "Mixed Valid/Invalid")
    assert isinstance(fig, go.Figure)

    stats = get_edge_display_stats(edges)
    assert isinstance(stats, dict)
    assert stats["valid_pixels"] < stats["total_pixels"]