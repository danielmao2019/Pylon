"""Tests for image display statistics functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch

from data.viewer.utils.atomic_displays.image_display import (
    get_image_display_stats
)


# ================================================================================
# get_image_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_image_display_stats_invalid_image_type():
    """Test assertion failure for invalid image input type."""
    with pytest.raises(AssertionError) as exc_info:
        get_image_display_stats("not_a_tensor")

    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_image_display_stats_invalid_image_dimensions():
    """Test assertion failure for invalid image dimensions."""
    image = torch.randn(32, 32, dtype=torch.float32)

    with pytest.raises(AssertionError) as exc_info:
        get_image_display_stats(image)

    assert "Expected 3D [C,H,W] or 4D [N,C,H,W] tensor" in str(exc_info.value)
