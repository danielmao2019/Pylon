"""Tests for segmentation display statistics functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.segmentation_display import (
    get_segmentation_display_stats
)


# ================================================================================
# get_segmentation_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_segmentation_display_stats_invalid_segmentation_type():
    """Test assertion failure for invalid segmentation input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats("not_valid_input")
    
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
    
    assert "cannot be empty" in str(exc_info.value)


def test_get_segmentation_display_stats_dict_missing_masks():
    """Test assertion failure for dict format missing masks key."""
    invalid_dict = {'indices': [0, 1, 2]}
    
    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(invalid_dict)
    
    assert "must have 'masks'" in str(exc_info.value)


def test_get_segmentation_display_stats_dict_missing_indices():
    """Test assertion failure for dict format missing indices key."""
    masks = [torch.ones(32, 32, dtype=torch.bool)]
    invalid_dict = {'masks': masks}
    
    with pytest.raises(AssertionError) as exc_info:
        get_segmentation_display_stats(invalid_dict)
    
    assert "must have 'indices'" in str(exc_info.value)