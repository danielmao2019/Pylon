"""
String representation tests for TorchvisionWrapper.

Tests the __str__ method to ensure proper formatting without wrapper prefix.
"""
import torch
import torchvision.transforms as T
from data.transforms.torchvision_wrapper import TorchvisionWrapper


def test_string_representation_with_parameters():
    """Test string representation with transform parameters."""
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.5, contrast=0.3, saturation=0.2)
    str_repr = str(wrapper)

    # Should show only the inner transform, not the wrapper
    assert "ColorJitter(" in str_repr
    assert "brightness=0.5" in str_repr
    assert "contrast=0.3" in str_repr
    assert "saturation=0.2" in str_repr

    # Should NOT contain wrapper information
    assert "TorchvisionWrapper(" not in str_repr


def test_string_representation_without_parameters():
    """Test string representation without parameters."""
    wrapper = TorchvisionWrapper(T.RandomHorizontalFlip)
    str_repr = str(wrapper)

    # Should show only the inner transform
    assert str_repr == "RandomHorizontalFlip()"

    # Should NOT contain wrapper information
    assert "TorchvisionWrapper(" not in str_repr


def test_string_representation_single_parameter():
    """Test string representation with single parameter."""
    wrapper = TorchvisionWrapper(T.RandomRotation, degrees=30)
    str_repr = str(wrapper)

    assert "RandomRotation(" in str_repr
    assert "degrees=30" in str_repr
    assert "TorchvisionWrapper(" not in str_repr


def test_string_representation_complex_parameters():
    """Test string representation with complex parameter types."""
    wrapper = TorchvisionWrapper(
        T.RandomAffine,
        degrees=(-15, 15),
        translate=(0.1, 0.2),
        scale=(0.9, 1.1)
    )
    str_repr = str(wrapper)

    assert "RandomAffine(" in str_repr
    assert "degrees=(-15, 15)" in str_repr
    assert "translate=(0.1, 0.2)" in str_repr
    assert "scale=(0.9, 1.1)" in str_repr
    assert "TorchvisionWrapper(" not in str_repr


def test_string_representation_boolean_parameters():
    """Test string representation with boolean parameters."""
    wrapper = TorchvisionWrapper(T.RandomHorizontalFlip, p=0.7)
    str_repr = str(wrapper)

    assert "RandomHorizontalFlip(" in str_repr
    assert "p=0.7" in str_repr
    assert "TorchvisionWrapper(" not in str_repr


def test_string_representation_format_consistency():
    """Test that string representation format is consistent with BaseTransform.format_params."""
    wrapper1 = TorchvisionWrapper(T.ColorJitter, brightness=0.5, contrast=0.5)
    wrapper2 = TorchvisionWrapper(T.RandomAffine, degrees=10)

    str1 = str(wrapper1)
    str2 = str(wrapper2)

    # Both should follow the pattern: TransformName(param=value, ...)
    assert str1.startswith("ColorJitter(") and str1.endswith(")")
    assert str2.startswith("RandomAffine(") and str2.endswith(")")

    # Should contain properly formatted parameters
    assert ", " in str1  # Multiple parameters separated by comma and space
    assert "=" in str1   # Parameter assignment format
    assert "=" in str2   # Parameter assignment format


def test_string_representation_different_transforms():
    """Test string representation for various transform types."""
    test_cases = [
        (T.ColorJitter, {"brightness": 0.4}, "ColorJitter"),
        (T.RandomRotation, {"degrees": 45}, "RandomRotation"),
        (T.RandomAffine, {"degrees": 10}, "RandomAffine"),
        (T.RandomHorizontalFlip, {"p": 0.5}, "RandomHorizontalFlip"),
    ]

    for transform_class, kwargs, expected_name in test_cases:
        wrapper = TorchvisionWrapper(transform_class, **kwargs)
        str_repr = str(wrapper)

        assert str_repr.startswith(expected_name + "("), f"String representation should start with {expected_name}("
        assert str_repr.endswith(")"), "String representation should end with )"
        assert "TorchvisionWrapper" not in str_repr, "Should not contain wrapper name"