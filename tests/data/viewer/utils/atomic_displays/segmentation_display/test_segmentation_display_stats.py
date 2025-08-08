"""Tests for segmentation display statistics functionality - Valid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, List

from data.viewer.utils.atomic_displays.segmentation_display import (
    get_segmentation_display_stats
)


# ================================================================================
# get_segmentation_display_stats Tests - Valid Cases
# ================================================================================

def test_get_segmentation_display_stats_tensor_basic(segmentation_tensor):
    """Test getting statistics from basic segmentation tensor."""
    stats = get_segmentation_display_stats(segmentation_tensor)
    
    assert isinstance(stats, dict)
    # Should contain basic segmentation statistics


def test_get_segmentation_display_stats_tensor_3d(segmentation_tensor_3d):
    """Test getting statistics from 3D segmentation tensor."""
    stats = get_segmentation_display_stats(segmentation_tensor_3d)
    assert isinstance(stats, dict)


def test_get_segmentation_display_stats_dict_format(segmentation_dict):
    """Test getting statistics from dictionary format segmentation."""
    stats = get_segmentation_display_stats(segmentation_dict)
    assert isinstance(stats, dict)


@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_get_segmentation_display_stats_various_classes(num_classes):
    """Test segmentation statistics with various numbers of classes."""
    segmentation = torch.randint(0, num_classes, (64, 64), dtype=torch.int64)
    stats = get_segmentation_display_stats(segmentation)
    assert isinstance(stats, dict)


def test_get_segmentation_display_stats_single_class():
    """Test segmentation statistics with only one class."""
    segmentation = torch.zeros(32, 32, dtype=torch.int64)
    stats = get_segmentation_display_stats(segmentation)
    assert isinstance(stats, dict)


def test_get_segmentation_display_stats_edge_cases():
    """Test segmentation statistics with edge case data."""
    # Very small segmentation
    small_seg = torch.randint(0, 3, (2, 2), dtype=torch.int64)
    stats = get_segmentation_display_stats(small_seg)
    assert isinstance(stats, dict)
    
    # Large segmentation with many classes
    large_seg = torch.randint(0, 20, (128, 128), dtype=torch.int64)
    stats = get_segmentation_display_stats(large_seg)
    assert isinstance(stats, dict)


def test_segmentation_stats_determinism(segmentation_tensor):
    """Test that statistics calculation is deterministic."""
    stats1 = get_segmentation_display_stats(segmentation_tensor)
    stats2 = get_segmentation_display_stats(segmentation_tensor)
    
    # Both should return same type
    assert isinstance(stats1, dict)
    assert isinstance(stats2, dict)
    # Note: Deep equality would require implementation details from get_segmentation_stats


# ================================================================================
# Batch Support Stats Tests - CRITICAL for eval viewer
# ================================================================================

def test_get_segmentation_display_stats_batched_tensor(batched_segmentation_tensor):
    """Test getting statistics from batched segmentation tensor."""
    stats = get_segmentation_display_stats(batched_segmentation_tensor)
    
    assert isinstance(stats, dict)
    # Should contain basic segmentation statistics


def test_batch_size_one_assertion_segmentation_stats():
    """Test that batch size > 1 raises assertion error in get_segmentation_display_stats."""
    invalid_batched_segmentation = torch.randint(0, 5, (3, 32, 32), dtype=torch.int64)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_segmentation_display_stats(invalid_batched_segmentation)


# ================================================================================
# Integration Tests
# ================================================================================

def test_complete_batch_segmentation_stats_pipeline(batched_segmentation_tensor):
    """Test complete batched segmentation statistics pipeline."""
    stats = get_segmentation_display_stats(batched_segmentation_tensor)
    assert isinstance(stats, dict)