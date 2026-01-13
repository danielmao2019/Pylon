"""Tests for segmentation display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any

from data.viewer.utils.atomic_displays.segmentation_display import (
    create_segmentation_display,
    get_segmentation_display_stats
)


# ================================================================================
# create_segmentation_display Tests - Invalid Cases
# ================================================================================

def test_create_segmentation_display_invalid_segmentation_type():
    """Test assertion failure for invalid segmentation input type."""
    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display("not_a_tensor_or_dict", "Test")

    assert "segmentation must be torch.Tensor or dict" in str(exc_info.value)


def test_create_segmentation_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    segmentation = torch.randint(0, 5, (32, 32), dtype=torch.int64)

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation, 123)

    assert "Expected str title" in str(exc_info.value)


def test_create_segmentation_display_tensor_invalid_dimensions():
    """Test assertion failure for tensor with invalid dimensions."""
    # 1D tensor
    segmentation_1d = torch.randint(0, 5, (100,), dtype=torch.int64)
    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_1d, "Test")
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)

    # 4D tensor
    segmentation_4d = torch.randint(0, 5, (1, 3, 32, 32), dtype=torch.int64)
    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_4d, "Test")
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)


def test_create_segmentation_display_tensor_invalid_batch_size():
    """Test assertion failure for 3D tensor with batch size > 1."""
    segmentation = torch.randint(0, 5, (3, 32, 32), dtype=torch.int64)

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation, "Test")

    assert "Expected batch size 1 for visualization" in str(exc_info.value)


def test_create_segmentation_display_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_seg = torch.empty((0, 0), dtype=torch.int64)

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(empty_seg, "Test")

    assert "Segmentation tensor cannot be empty" in str(exc_info.value)


def test_create_segmentation_display_tensor_invalid_dtype():
    """Test assertion failure for tensor with invalid dtype."""
    segmentation = torch.randint(0, 5, (32, 32), dtype=torch.float32)

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation, "Test")

    assert "Expected int64 segmentation" in str(exc_info.value)


def test_create_segmentation_display_dict_missing_masks():
    """Test assertion failure for dict missing 'masks' key."""
    segmentation_dict = {
        'indices': [0, 1, 2]
    }

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_dict, "Test")

    assert "Dict segmentation must have 'masks'" in str(exc_info.value)


def test_create_segmentation_display_dict_missing_indices():
    """Test assertion failure for dict missing 'indices' key."""
    masks = [torch.zeros(32, 32, dtype=torch.bool) for _ in range(3)]
    segmentation_dict = {
        'masks': masks
    }

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_dict, "Test")

    assert "Dict segmentation must have 'indices'" in str(exc_info.value)


def test_create_segmentation_display_dict_invalid_masks_type():
    """Test assertion failure for dict with invalid masks type."""
    segmentation_dict = {
        'masks': "not_a_list",
        'indices': [0, 1, 2]
    }

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_dict, "Test")

    assert "masks must be list" in str(exc_info.value)


def test_create_segmentation_display_dict_invalid_indices_type():
    """Test assertion failure for dict with invalid indices type."""
    masks = [torch.zeros(32, 32, dtype=torch.bool) for _ in range(3)]
    segmentation_dict = {
        'masks': masks,
        'indices': "not_a_list"
    }

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_dict, "Test")

    assert "indices must be list" in str(exc_info.value)


def test_create_segmentation_display_dict_empty_masks():
    """Test assertion failure for dict with empty masks."""
    segmentation_dict = {
        'masks': [],
        'indices': []
    }

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_dict, "Test")

    assert "masks cannot be empty" in str(exc_info.value)


def test_create_segmentation_display_dict_mismatched_lengths():
    """Test assertion failure for dict with mismatched masks and indices lengths."""
    masks = [torch.zeros(32, 32, dtype=torch.bool) for _ in range(3)]
    indices = [0, 1]  # Different length than masks

    segmentation_dict = {
        'masks': masks,
        'indices': indices
    }

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation_dict, "Test")

    assert "masks and indices must have same length" in str(exc_info.value)


def test_create_segmentation_display_invalid_class_labels_type():
    """Test assertion failure for invalid class_labels type."""
    segmentation = torch.randint(0, 5, (32, 32), dtype=torch.int64)

    with pytest.raises(AssertionError) as exc_info:
        create_segmentation_display(segmentation, "Test", class_labels="not_a_dict")

    assert "class_labels must be dict" in str(exc_info.value)


# ================================================================================
# get_segmentation_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_segmentation_display_stats_invalid_segmentation_type():
    """Test assertion failure for invalid segmentation type."""
    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats("not_a_tensor_or_dict")

    assert "segmentation must be torch.Tensor or dict" in str(exc_info.value)


def test_get_segmentation_display_stats_tensor_invalid_dimensions():
    """Test assertion failure for tensor with invalid dimensions."""
    # 1D tensor
    segmentation_1d = torch.randint(0, 5, (100,), dtype=torch.int64)
    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(segmentation_1d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)

    # 4D tensor
    segmentation_4d = torch.randint(0, 5, (1, 3, 32, 32), dtype=torch.int64)
    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(segmentation_4d)
    assert "Expected 2D [H,W] or 3D [N,H,W] tensor" in str(exc_info.value)


def test_get_segmentation_display_stats_empty_tensor():
    """Test assertion failure for empty tensor."""
    empty_seg = torch.empty((0, 0), dtype=torch.int64)

    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(empty_seg)

    assert "Segmentation tensor cannot be empty" in str(exc_info.value)


def test_get_segmentation_display_stats_dict_missing_masks():
    """Test assertion failure for dict missing 'masks' key."""
    segmentation_dict = {
        'indices': [0, 1, 2]
    }

    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(segmentation_dict)

    assert "Dict segmentation must have 'masks'" in str(exc_info.value)


def test_get_segmentation_display_stats_dict_missing_indices():
    """Test assertion failure for dict missing 'indices' key."""
    masks = [torch.zeros(32, 32, dtype=torch.bool) for _ in range(3)]
    segmentation_dict = {
        'masks': masks
    }

    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(segmentation_dict)

    assert "Dict segmentation must have 'indices'" in str(exc_info.value)


# ================================================================================
# Edge Cases and Boundary Testing
# ================================================================================

def test_create_segmentation_display_extreme_tensor_values():
    """Test with extreme tensor values that might cause issues."""
    # Very large class indices
    large_seg = torch.full((32, 32), 1000000, dtype=torch.int64)
    fig = create_segmentation_display(large_seg, "Large Values Test")
    assert isinstance(fig, go.Figure)

    # Negative class indices (should work but might be unusual)
    negative_seg = torch.full((32, 32), -1, dtype=torch.int64)
    fig = create_segmentation_display(negative_seg, "Negative Values Test")
    assert isinstance(fig, go.Figure)


def test_create_segmentation_display_dict_with_overlapping_masks():
    """Test dict format with overlapping masks (edge case)."""
    # Create overlapping masks
    mask1 = torch.zeros(32, 32, dtype=torch.bool)
    mask1[10:20, 10:20] = True

    mask2 = torch.zeros(32, 32, dtype=torch.bool)
    mask2[15:25, 15:25] = True  # Overlaps with mask1

    segmentation_dict = {
        'masks': [mask1, mask2],
        'indices': [0, 1]
    }

    # Should work even with overlapping masks
    fig = create_segmentation_display(segmentation_dict, "Overlapping Test")
    assert isinstance(fig, go.Figure)


def test_segmentation_display_with_none_values():
    """Test various None input scenarios."""
    segmentation = torch.randint(0, 5, (32, 32), dtype=torch.int64)

    # This should work (class_labels is optional)
    fig = create_segmentation_display(segmentation, "Test", class_labels=None)
    assert isinstance(fig, go.Figure)