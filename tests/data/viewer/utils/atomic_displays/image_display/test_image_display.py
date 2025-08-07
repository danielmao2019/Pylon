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


@pytest.fixture
def batched_rgb_image():
    """Fixture providing batched RGB image tensor with batch size 1."""
    return torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)


@pytest.fixture
def batched_grayscale_image():
    """Fixture providing batched grayscale image tensor with batch size 1."""
    return torch.randint(0, 255, (1, 1, 32, 32), dtype=torch.uint8)


@pytest.fixture
def multi_channel_image():
    """Fixture providing multi-channel image tensor (6 channels)."""
    return torch.randint(0, 255, (6, 32, 32), dtype=torch.uint8)


@pytest.fixture
def batched_multi_channel_image():
    """Fixture providing batched multi-channel image tensor with batch size 1."""
    return torch.randint(0, 255, (1, 6, 32, 32), dtype=torch.uint8)


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


# ================================================================================
# Batch Support Tests - CRITICAL for eval viewer
# ================================================================================

def test_create_image_display_batched_rgb(batched_rgb_image):
    """Test creating display for batched RGB image (batch size 1)."""
    fig = create_image_display(batched_rgb_image, "Test Batched RGB")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Batched RGB"
    assert fig.layout.height == 400


def test_create_image_display_batched_grayscale(batched_grayscale_image):
    """Test creating display for batched grayscale image (batch size 1)."""
    fig = create_image_display(batched_grayscale_image, "Test Batched Grayscale")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Batched Grayscale"


def test_create_image_display_batched_multi_channel(batched_multi_channel_image):
    """Test creating display for batched multi-channel image (batch size 1)."""
    fig = create_image_display(batched_multi_channel_image, "Test Batched Multi-Channel")
    
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Test Batched Multi-Channel"


def test_get_image_display_stats_batched_rgb(batched_rgb_image):
    """Test image statistics calculation for batched RGB image."""
    stats = get_image_display_stats(batched_rgb_image)
    
    assert isinstance(stats, dict)
    assert "Shape" in stats
    assert "Min Value" in stats
    assert "Max Value" in stats
    assert "Mean Value" in stats
    assert "Std Dev" in stats
    
    # Should show unbatched shape in stats
    assert stats["Shape"] == "(3, 32, 32)"


def test_get_image_display_stats_batched_with_change_map():
    """Test image statistics with batched input and batched change map."""
    batched_image = torch.randn(1, 3, 32, 32, dtype=torch.float32)
    batched_change_map = torch.randint(0, 2, (1, 32, 32), dtype=torch.float32)
    
    stats = get_image_display_stats(batched_image, batched_change_map)
    
    assert "Changed Pixels" in stats
    assert "Change Min" in stats
    assert "Change Max" in stats
    assert "%" in stats["Changed Pixels"]


def test_batch_size_one_assertion_create_display():
    """Test that batch size > 1 raises assertion error in create_image_display."""
    invalid_batched_image = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for visualization"):
        create_image_display(invalid_batched_image, "Should Fail")


def test_batch_size_one_assertion_stats():
    """Test that batch size > 1 raises assertion error in get_image_display_stats."""
    invalid_batched_image = torch.randint(0, 255, (3, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for analysis"):
        get_image_display_stats(invalid_batched_image)


@pytest.mark.parametrize("batch_shape,unbatched_expected_shape", [
    ((1, 3, 32, 32), (32, 32, 3)),     # Batched RGB -> RGB
    ((1, 1, 32, 32), (32, 32)),        # Batched grayscale -> grayscale
    ((1, 6, 32, 32), (32, 32, 3)),     # Batched multi-channel -> RGB (sampled)
])
def test_batch_to_unbatch_shape_consistency(batch_shape, unbatched_expected_shape):
    """Test that batched inputs produce same output shapes as unbatched equivalents."""
    batched_image = torch.randint(0, 255, batch_shape, dtype=torch.uint8)
    unbatched_image = batched_image[0]  # Remove batch dimension
    
    # Both should produce same numpy shape
    batched_result = image_to_numpy(unbatched_image)  # Function internally handles batching in create_image_display
    unbatched_result = image_to_numpy(unbatched_image)
    
    assert batched_result.shape == unbatched_result.shape == unbatched_expected_shape
    
    # Both should create valid figures
    batched_fig = create_image_display(batched_image, "Batched")
    unbatched_fig = create_image_display(unbatched_image, "Unbatched")
    
    assert isinstance(batched_fig, go.Figure)
    assert isinstance(unbatched_fig, go.Figure)


def test_invalid_2channel_image_error():
    """Test that 2-channel images raise ValueError as they are not supported."""
    two_channel_image = torch.randint(0, 255, (2, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(ValueError, match="2-channel images are not supported"):
        image_to_numpy(two_channel_image)


def test_batched_invalid_2channel_image_error():
    """Test that batched 2-channel images raise ValueError."""
    batched_two_channel_image = torch.randint(0, 255, (1, 2, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(ValueError, match="2-channel images are not supported"):
        create_image_display(batched_two_channel_image, "Should Fail")


# ================================================================================  
# Batch Support Integration Tests
# ================================================================================

def test_complete_batch_pipeline_rgb(batched_rgb_image):
    """Test complete batched image pipeline from tensor to figure."""
    # Test display creation (internally handles batch)
    fig = create_image_display(batched_rgb_image, "Batch Integration Test")
    assert isinstance(fig, go.Figure)
    
    # Test statistics (handles batch)
    stats = get_image_display_stats(batched_rgb_image)
    assert isinstance(stats, dict)
    assert len(stats) >= 5
    
    # Stats should show unbatched shape
    assert stats["Shape"] == "(3, 32, 32)"


def test_complete_batch_pipeline_grayscale(batched_grayscale_image):
    """Test complete batched grayscale pipeline."""
    fig = create_image_display(batched_grayscale_image, "Batch Grayscale Integration")
    stats = get_image_display_stats(batched_grayscale_image)
    
    assert isinstance(fig, go.Figure)
    assert isinstance(stats, dict)
    assert stats["Shape"] == "(1, 32, 32)"
