"""Tests for image display functionality - Valid Cases.

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
# image_to_numpy Tests - Valid Cases
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


# ================================================================================
# create_image_display Tests - Valid Cases
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


# ================================================================================
# get_image_display_stats Tests - Valid Cases
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
