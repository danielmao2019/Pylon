"""Tests for image display functionality - Invalid Cases.

CRITICAL: Uses pytest FUNCTIONS only (no test classes) as required by CLAUDE.md.
"""
import pytest
import torch


from data.viewer.utils.atomic_displays.image_display import (
    _image_to_numpy,
    create_image_display
)


# ================================================================================
# image_to_numpy Tests - Invalid Cases
# ================================================================================

def test_image_to_numpy_invalid_input_type():
    """Test assertion failure for invalid input type."""
    with pytest.raises(AssertionError) as exc_info:
        _image_to_numpy("not_a_tensor")
    
    assert "Expected torch.Tensor" in str(exc_info.value)


def test_image_to_numpy_invalid_batch_size():
    """Test assertion failure for 4D input (image_to_numpy only accepts 3D)."""
    image = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        _image_to_numpy(image)
    
    assert "Expected 3D tensor [C,H,W]" in str(exc_info.value)


def test_image_to_numpy_invalid_shape():
    """Test handling of invalid tensor shapes."""
    image = torch.randint(0, 255, (100,), dtype=torch.uint8)
    
    with pytest.raises(AssertionError) as exc_info:
        _image_to_numpy(image)
    
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


def test_batch_size_one_assertion_create_display():
    """Test that batch size > 1 raises assertion error in create_image_display."""
    invalid_batched_image = torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(AssertionError, match="Expected batch size 1 for visualization"):
        create_image_display(invalid_batched_image, "Should Fail")


def test_invalid_2channel_image_error():
    """Test that 2-channel images raise ValueError as they are not supported."""
    two_channel_image = torch.randint(0, 255, (2, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(ValueError, match="2-channel images are not supported"):
        _image_to_numpy(two_channel_image)


def test_batched_invalid_2channel_image_error():
    """Test that batched 2-channel images raise ValueError."""
    batched_two_channel_image = torch.randint(0, 255, (1, 2, 32, 32), dtype=torch.uint8)
    
    with pytest.raises(ValueError, match="2-channel images are not supported"):
        create_image_display(batched_two_channel_image, "Should Fail")
