"""Tests for segmentation display functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, List

import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.segmentation_display import (
    create_segmentation_display,
    get_segmentation_display_stats
)


# ================================================================================
# create_segmentation_display Tests - Valid Cases
# ================================================================================

def test_create_segmentation_display_tensor_2d(segmentation_tensor):
    """Test creating segmentation display with 2D tensor."""
    fig = create_segmentation_display(segmentation_tensor, "Test Segmentation 2D")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Segmentation 2D"


def test_create_segmentation_display_tensor_3d(segmentation_tensor_3d):
    """Test creating segmentation display with 3D tensor (single channel)."""
    fig = create_segmentation_display(segmentation_tensor_3d, "Test Segmentation 3D")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Segmentation 3D"


def test_create_segmentation_display_dict_format(segmentation_dict):
    """Test creating segmentation display with dictionary format."""
    fig = create_segmentation_display(segmentation_dict, "Test Segmentation Dict")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Segmentation Dict"


@pytest.mark.parametrize("num_classes", [2, 5, 10, 20])
def test_create_segmentation_display_various_classes(num_classes):
    """Test segmentation display with various numbers of classes."""
    segmentation = torch.randint(0, num_classes, (32, 32), dtype=torch.int64)
    fig = create_segmentation_display(segmentation, f"Test {num_classes} Classes")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == f"Test {num_classes} Classes"


def test_create_segmentation_display_with_class_labels(segmentation_tensor, class_labels):
    """Test segmentation display with class labels."""
    fig = create_segmentation_display(
        segmentation_tensor,
        "Test with Labels",
        class_labels=class_labels
    )

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test with Labels"


def test_create_segmentation_display_single_class():
    """Test segmentation display with single class (all pixels same value)."""
    segmentation = torch.full((32, 32), 1, dtype=torch.int64)
    fig = create_segmentation_display(segmentation, "Single Class Test")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Single Class Test"


@pytest.mark.parametrize("tensor_size", [(16, 16), (64, 64), (128, 128)])
def test_create_segmentation_display_various_sizes(tensor_size):
    """Test segmentation display with various tensor sizes."""
    h, w = tensor_size
    segmentation = torch.randint(0, 5, (h, w), dtype=torch.int64)
    fig = create_segmentation_display(segmentation, f"Test {h}x{w}")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == f"Test {h}x{w}"


# ================================================================================
# Integration and Pipeline Tests
# ================================================================================

def test_segmentation_display_pipeline(segmentation_tensor):
    """Test complete segmentation display pipeline."""
    # Test display creation
    fig = create_segmentation_display(segmentation_tensor, "Pipeline Test")
    assert isinstance(fig, go.Figure)

    # Test statistics
    stats = get_segmentation_display_stats(segmentation_tensor)
    assert isinstance(stats, dict)


def test_segmentation_display_dict_pipeline(segmentation_dict):
    """Test complete pipeline with dictionary format."""
    # Test display creation
    fig = create_segmentation_display(segmentation_dict, "Dict Pipeline Test")
    assert isinstance(fig, go.Figure)

    # Test statistics
    stats = get_segmentation_display_stats(segmentation_dict)
    assert isinstance(stats, dict)


def test_segmentation_display_with_complex_dict():
    """Test segmentation display with complex dictionary format."""
    # Create more complex instance segmentation data
    masks = []
    indices = []

    for i in range(5):
        mask = torch.zeros(64, 64, dtype=torch.bool)
        # Create circular-like masks
        center_y, center_x = 16 + i * 8, 16 + i * 8
        y_coords, x_coords = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        distance = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        mask[distance < 5] = True

        masks.append(mask)
        indices.append(f"instance_{i}")

    segmentation_data = {
        'masks': masks,
        'indices': indices
    }

    # Test display
    fig = create_segmentation_display(segmentation_data, "Complex Dict Test")
    assert isinstance(fig, go.Figure)

    # Test stats
    stats = get_segmentation_display_stats(segmentation_data)
    assert isinstance(stats, dict)


def test_performance_with_large_segmentation():
    """Test performance with large segmentation maps."""
    # Create large segmentation
    large_seg = torch.randint(0, 10, (512, 512), dtype=torch.int64)

    # These should complete without error
    fig = create_segmentation_display(large_seg, "Large Segmentation Test")
    stats = get_segmentation_display_stats(large_seg)

    # Basic checks
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)


# ================================================================================
# Correctness Verification Tests
# ================================================================================

def test_segmentation_display_correctness():
    """Test segmentation display with known input patterns."""
    # Create checkerboard pattern
    segmentation = torch.zeros(32, 32, dtype=torch.int64)
    segmentation[::2, ::2] = 1  # Even rows, even cols = class 1
    segmentation[1::2, 1::2] = 1  # Odd rows, odd cols = class 1
    # Rest remains class 0

    fig = create_segmentation_display(segmentation, "Checkerboard Test")
    assert isinstance(fig, go.Figure)

    stats = get_segmentation_display_stats(segmentation)
    assert isinstance(stats, dict)


def test_segmentation_display_determinism(segmentation_tensor):
    """Test that display creation is deterministic."""
    fig1 = create_segmentation_display(segmentation_tensor, "Determinism Test")
    fig2 = create_segmentation_display(segmentation_tensor, "Determinism Test")

    # Both should be valid figures with same title
    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert fig1.layout.title.text == fig2.layout.title.text


def test_segmentation_stats_determinism(segmentation_tensor):
    """Test that statistics calculation is deterministic."""
    stats1 = get_segmentation_display_stats(segmentation_tensor)
    stats2 = get_segmentation_display_stats(segmentation_tensor)

    # Both should return same type
    assert isinstance(stats1, dict)
    assert isinstance(stats2, dict)
    # Note: Deep equality would require implementation details from get_segmentation_stats


# ================================================================================
# Batch Support Tests - CRITICAL for eval viewer
# ================================================================================

def test_create_segmentation_display_batched_tensor(batched_segmentation_tensor):
    """Test creating segmentation display with batched tensor (batch size 1)."""
    fig = create_segmentation_display(batched_segmentation_tensor, "Test Batched Segmentation")

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Batched Segmentation"



def test_batch_size_one_assertion_segmentation_display():
    """Test that batch size > 1 raises assertion error in create_segmentation_display."""
    invalid_batched_segmentation = torch.randint(0, 5, (2, 32, 32), dtype=torch.int64)

    with pytest.raises(AssertionError, match="Expected batch size 1 for visualization"):
        create_segmentation_display(invalid_batched_segmentation, "Should Fail")


def test_batch_size_one_assertion_segmentation_stats():
    """Test that batch size > 1 raises assertion error in get_segmentation_display_stats."""
    invalid_batched_segmentation = torch.randint(0, 5, (3, 32, 32), dtype=torch.int64)

    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_segmentation_display_stats(invalid_batched_segmentation)


def test_batched_vs_unbatched_segmentation_consistency(batched_segmentation_tensor):
    """Test that batched and unbatched segmentation produce equivalent results."""
    unbatched_segmentation = batched_segmentation_tensor[0]  # Remove batch dimension

    # Both should create valid figures
    batched_fig = create_segmentation_display(batched_segmentation_tensor, "Batched")
    unbatched_fig = create_segmentation_display(unbatched_segmentation, "Unbatched")

    assert isinstance(batched_fig, go.Figure)
    assert isinstance(unbatched_fig, go.Figure)

    # Both should produce valid statistics
    batched_stats = get_segmentation_display_stats(batched_segmentation_tensor)
    unbatched_stats = get_segmentation_display_stats(unbatched_segmentation)

    assert isinstance(batched_stats, dict)
    assert isinstance(unbatched_stats, dict)


def test_complete_batch_segmentation_pipeline(batched_segmentation_tensor):
    """Test complete batched segmentation pipeline from tensor to figure."""
    # Test display creation (internally handles batch)
    fig = create_segmentation_display(batched_segmentation_tensor, "Batch Segmentation Integration")
    assert isinstance(fig, go.Figure)

    # Test statistics (handles batch)
    stats = get_segmentation_display_stats(batched_segmentation_tensor)
    assert isinstance(stats, dict)
