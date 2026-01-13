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
