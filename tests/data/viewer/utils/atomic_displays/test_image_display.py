"""Tests for image display functionality.

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
# Fixtures
# ================================================================================

@pytest.fixture
def rgb_image():
    """Fixture providing RGB image tensor."""
    return torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)


@pytest.fixture  
def grayscale_image():
    """Fixture providing grayscale image tensor."""
    return torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8)


# ================================================================================
# image_to_numpy Tests
# ================================================================================

def test_image_to_numpy_rgb_tensor(rgb_image):
    """Test converting RGB image tensor to numpy."""
    result = image_to_numpy(rgb_image)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32, 3)
    assert result.dtype == np.float64


def test_image_to_numpy_grayscale_tensor(grayscale_image):
    """Test converting grayscale image tensor to numpy."""
    result = image_to_numpy(grayscale_image)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32)
    assert result.dtype == np.float64


@pytest.mark.parametrize("input_shape,expected_output_shape", [
    ((3, 32, 32), (32, 32, 3)),  # RGB
    ((1, 32, 32), (32, 32)),     # Grayscale
    ((6, 32, 32), (32, 32, 3)),  # Many channels (randomly sampled)
])
def test_image_to_numpy_shapes(input_shape, expected_output_shape):
    """Test image_to_numpy with various input shapes."""
    image = torch.randint(0, 255, input_shape, dtype=torch.uint8)
    result = image_to_numpy(image)
    assert result.shape == expected_output_shape


def test_image_to_numpy_normalization():
    """Test that image values are normalized to [0, 1]."""
    image = torch.tensor([[[100, 200], [50, 255]]], dtype=torch.uint8)
    result = image_to_numpy(image)
    
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_image_to_numpy_uniform_values():
    """Test handling of uniform pixel values (avoid division by zero)."""
    image = torch.full((3, 32, 32), 128, dtype=torch.uint8)
    result = image_to_numpy(image)
    
    assert isinstance(result, np.ndarray)
    assert np.all(result == 0.0)


def test_image_to_numpy_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        image_to_numpy("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_image_to_numpy_invalid_batch_size():
    """Test assertion failure for invalid batch size."""
    image = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        image_to_numpy(image)
    
    assert "Expected batch size 1" in str(exc_info.value)


def test_image_to_numpy_invalid_shape():
    """Test handling of invalid tensor shapes."""
    image = torch.randint(0, 255, (100,), dtype=torch.uint8)
    
    with pytest.raises(ValueError) as exc_info:
        image_to_numpy(image)
    
    assert "Unsupported tensor shape" in str(exc_info.value)


# ================================================================================
# create_image_display Tests
# ================================================================================

def test_create_image_display_rgb(rgb_image):
    """Test creating display for RGB image."""
    fig = create_image_display(rgb_image, "Test RGB Image")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test RGB Image"
    assert fig.layout.height == 400


def test_create_image_display_grayscale(grayscale_image):
    """Test creating display for grayscale image."""
    fig = create_image_display(grayscale_image, "Test Grayscale Image")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Grayscale Image"


@pytest.mark.parametrize("colorscale", ["Viridis", "Plasma", "Inferno", "Cividis"])
def test_create_image_display_various_colorscales(rgb_image, colorscale):
    """Test creating display with various colorscales."""
    fig = create_image_display(rgb_image, "Test", colorscale=colorscale)
    assert isinstance(fig, go.Figure)


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
    
    assert "Expected 3D tensor [C,H,W]" in str(exc_info.value)


def test_create_image_display_invalid_channels():
    """Test assertion failure for invalid number of channels."""
    image = torch.randint(0, 255, (2, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        create_image_display(image, "Test")
    
    assert "Expected 1 or 3 channels" in str(exc_info.value)


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
# get_image_display_stats Tests
# ================================================================================

def test_get_image_display_stats_basic():
    """Test basic image statistics calculation."""
    image = torch.randn(3, 32, 32, dtype=torch.float32)
    stats = get_image_display_stats(image)
    
    assert isinstance(stats, dict)
    assert "Shape" in stats
    assert "Min Value" in stats
    assert "Max Value" in stats
    assert "Mean Value" in stats
    assert "Std Dev" in stats
    
    assert stats["Shape"] == "(3, 32, 32)"


def test_get_image_display_stats_with_binary_change_map():
    """Test image statistics with binary change map."""
    image = torch.randn(3, 32, 32, dtype=torch.float32)
    change_map = torch.randint(0, 2, (32, 32), dtype=torch.float32)
    
    stats = get_image_display_stats(image, change_map)
    
    assert "Changed Pixels" in stats
    assert "Change Min" in stats
    assert "Change Max" in stats
    assert "%" in stats["Changed Pixels"]


def test_get_image_display_stats_with_multiclass_change_map():
    """Test image statistics with multi-class change map."""
    image = torch.randn(3, 32, 32, dtype=torch.float32)
    change_map = torch.randn(5, 32, 32, dtype=torch.float32)
    
    stats = get_image_display_stats(image, change_map)
    
    assert "Number of Classes" in stats
    assert "Class Distribution" in stats
    assert stats["Number of Classes"] == 5
    assert isinstance(stats["Class Distribution"], dict)


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
    
    assert "Expected 3D tensor [C,H,W]" in str(exc_info.value)


def test_get_image_display_stats_invalid_change_map_type():
    """Test assertion failure for invalid change_map type."""
    image = torch.randn(3, 32, 32, dtype=torch.float32)
    
    with pytest.raises(AssertionError) as exc_info:
        get_image_display_stats(image, "not_a_tensor")
    
    assert "change_map must be torch.Tensor" in str(exc_info.value)


# ================================================================================
# Integration and Edge Case Tests
# ================================================================================

def test_image_display_with_extreme_values():
    """Test image display handles extreme tensor values correctly."""
    # Very large values
    large_image = torch.full((3, 32, 32), 1e6, dtype=torch.float32)
    fig = create_image_display(large_image, "Large Values")
    assert isinstance(fig, go.Figure)
    
    # Very small values  
    small_image = torch.full((3, 32, 32), 1e-6, dtype=torch.float32)
    fig = create_image_display(small_image, "Small Values")
    assert isinstance(fig, go.Figure)
    
    # Mixed positive/negative
    mixed_image = torch.randn(3, 32, 32, dtype=torch.float32) * 1000
    fig = create_image_display(mixed_image, "Mixed Values")
    assert isinstance(fig, go.Figure)


def test_image_stats_with_edge_cases():
    """Test image statistics with edge case tensors."""
    # All zeros
    zero_image = torch.zeros(3, 32, 32, dtype=torch.float32)
    stats = get_image_display_stats(zero_image)
    assert float(stats["Min Value"]) == 0.0
    assert float(stats["Max Value"]) == 0.0
    
    # Single pixel
    tiny_image = torch.ones(3, 1, 1, dtype=torch.float32)
    stats = get_image_display_stats(tiny_image)
    assert stats["Shape"] == "(3, 1, 1)"


def test_image_display_pipeline(rgb_image):
    """Test complete image display pipeline from tensor to figure."""
    # Test conversion
    numpy_img = image_to_numpy(rgb_image)
    assert isinstance(numpy_img, np.ndarray)
    assert numpy_img.shape == (32, 32, 3)
    
    # Test display creation
    fig = create_image_display(rgb_image, "Integration Test")
    assert isinstance(fig, go.Figure)
    
    # Test statistics
    stats = get_image_display_stats(rgb_image)
    assert isinstance(stats, dict)
    assert len(stats) >= 5


def test_performance_with_large_images():
    """Test that image processing functions perform reasonably for large images."""
    # Create large image
    large_image = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    
    # These should complete without error or excessive time
    numpy_img = image_to_numpy(large_image)
    fig = create_image_display(large_image, "Large Image Test")
    stats = get_image_display_stats(large_image)
    
    # Basic checks
    assert numpy_img.shape == (512, 512, 3)
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)