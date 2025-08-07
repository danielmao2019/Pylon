"""Tests for image display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional

import plotly.graph_objects as go

from data.viewer.utils.atomic_displays.image_display import (
    image_to_numpy,
    create_image_display,
    get_image_display_stats
)


# ================================================================================
# image_to_numpy Tests - Invalid Cases
# ================================================================================

def test_image_to_numpy_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        image_to_numpy("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_image_to_numpy_invalid_batch_size():
    """Test assertion failure for 4D input (image_to_numpy only accepts 3D)."""
    image = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        image_to_numpy(image)
    
    assert "Expected 3D tensor [C,H,W]" in str(exc_info.value)


def test_image_to_numpy_invalid_shape():
    """Test handling of invalid tensor shapes."""
    image = torch.randint(0, 255, (100,), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        image_to_numpy(image)
    
    assert "Expected 3D tensor [C,H,W]" in str(exc_info.value)


# ================================================================================
# create_image_display Tests - Invalid Cases
# ================================================================================

def test_create_image_display_invalid_tensor_type():
    """Test assertion failure for invalid tensor input."""
    with pytest.raises(AssertionError) as exc_info:
        create_image_display("not_a_tensor", "Test")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_create_image_display_invalid_dimensions():
    """Test assertion failure for wrong tensor dimensions."""
    image = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        create_image_display(image, "Test")
    
    assert "Expected 3D [C,H,W] or 4D [N,C,H,W] tensor" in str(exc_info.value)


def test_create_image_display_invalid_channels():
    """Test ValueError for 2-channel images (not supported)."""
    image = torch.randint(0, 255, (2, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(ValueError) as exc_info:
        create_image_display(image, "Test")
    
    assert "2-channel images are not supported" in str(exc_info.value)


def test_create_image_display_empty_tensor():
    """Test assertion failure for empty tensor."""
    image = torch.empty((3, 0, 0), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        create_image_display(image, "Test")
    
    assert "Image tensor cannot be empty" in str(exc_info.value)


def test_create_image_display_invalid_title_type():
    """Test assertion failure for invalid title type."""
    image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        create_image_display(image, 123)
    
    assert "Expected str title" in str(exc_info.value)


def test_create_image_display_invalid_colorscale_type():
    """Test assertion failure for invalid colorscale type."""
    image = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        create_image_display(image, "Test", colorscale=123)
    
    assert "Expected str colorscale" in str(exc_info.value)


# ================================================================================
# get_image_display_stats Tests - Invalid Cases
# ================================================================================

def test_get_image_display_stats_invalid_image_type():
    """Test assertion failure for invalid image type."""
    with pytest.raises(AssertionError) as exc_info:
        get_image_display_stats("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_get_image_display_stats_invalid_image_dimensions():
    """Test assertion failure for invalid image dimensions."""
    image = torch.randn(32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_image_display_stats(image)
    
    assert "Expected 3D [C,H,W] or 4D [N,C,H,W] tensor" in str(exc_info.value)


def test_get_image_display_stats_invalid_change_map_type():
    """Test assertion failure for invalid change_map type."""
    image = torch.randn(3, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_image_display_stats(image, "not_a_tensor")
    
    assert "change_map must be torch.Tensor" in str(exc_info.value)
