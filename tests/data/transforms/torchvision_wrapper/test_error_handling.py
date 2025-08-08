"""
Error handling and input validation tests for TorchvisionWrapper.

Tests assertion-based input validation following Pylon's design philosophy.
"""
import torch
import torchvision.transforms as T
import pytest
from data.transforms.torchvision_wrapper import TorchvisionWrapper


def test_non_torchvision_class_assertion():
    """Test that non-torchvision class raises assertion error."""
    class NotTorchvisionTransform:
        def __init__(self):
            pass
    
    with pytest.raises(AssertionError, match="transform_class must be from torchvision"):
        TorchvisionWrapper(NotTorchvisionTransform)


def test_none_transform_class_assertion():
    """Test that None transform_class raises assertion error."""
    with pytest.raises(AssertionError, match="transform_class must not be None"):
        TorchvisionWrapper(None)


def test_non_callable_transform_class_assertion():
    """Test that non-callable transform_class raises assertion error."""
    not_callable = "not_a_class"
    
    # The assertion will fail on __module__ check first for string types
    with pytest.raises(AssertionError, match="transform_class must have __module__ attribute"):
        TorchvisionWrapper(not_callable)


def test_invalid_input_tensor_assertion():
    """Test that invalid input to _call_single raises assertion error."""
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.5)
    
    # Test with non-tensor input
    with pytest.raises(AssertionError, match="image must be torch.Tensor"):
        wrapper._call_single("not_a_tensor")
    
    # Test with non-tensor input (different type)
    with pytest.raises(AssertionError, match="image must be torch.Tensor"):
        wrapper._call_single([1, 2, 3])


def test_empty_tensor_assertion():
    """Test that empty tensor raises assertion error."""
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.5)
    
    # Test with empty tensor
    empty_tensor = torch.empty(0)
    with pytest.raises(AssertionError, match="image tensor must not be empty"):
        wrapper._call_single(empty_tensor)


def test_valid_torchvision_transform_acceptance():
    """Test that valid torchvision transforms are accepted without error."""
    # These should not raise any assertions
    wrapper1 = TorchvisionWrapper(T.ColorJitter, brightness=0.5)
    wrapper2 = TorchvisionWrapper(T.RandomAffine, degrees=10)
    wrapper3 = TorchvisionWrapper(T.RandomRotation, degrees=30)
    wrapper4 = TorchvisionWrapper(T.RandomHorizontalFlip)
    
    assert wrapper1.transform_class == T.ColorJitter
    assert wrapper2.transform_class == T.RandomAffine
    assert wrapper3.transform_class == T.RandomRotation
    assert wrapper4.transform_class == T.RandomHorizontalFlip


def test_valid_tensor_input_acceptance():
    """Test that valid tensor inputs are accepted without error."""
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.3)
    
    # Various valid tensor shapes should work
    valid_tensors = [
        torch.rand(3, 32, 32),  # Normal image
        torch.rand(1, 28, 28),  # Grayscale
        torch.rand(3, 64, 64),  # Different size
        torch.ones(3, 16, 16),  # All ones
        torch.zeros(3, 8, 8) + 0.1,  # Near zeros
    ]
    
    for tensor in valid_tensors:
        # Should not raise any assertions
        result = wrapper._call_single(tensor.clone(), None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape