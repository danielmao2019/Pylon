"""
Integration tests for TorchvisionWrapper with other Pylon components.

Tests integration with Randomize transform and other Pylon transform pipeline components.
"""
import torch
import torchvision.transforms as T
from data.transforms.torchvision_wrapper import TorchvisionWrapper
from data.transforms.randomize import Randomize


def test_integration_with_randomize_transform():
    """Test TorchvisionWrapper integration with Randomize transform."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 32, 32)

    # Create wrapped ColorJitter with Randomize
    wrapped_transform = TorchvisionWrapper(T.ColorJitter, brightness=0.5, contrast=0.5)
    randomize_transform = Randomize(transform=wrapped_transform, p=1.0)  # Always apply

    # Test with same seed
    result1 = randomize_transform(sample_image.clone(), seed=789)
    result2 = randomize_transform(sample_image.clone(), seed=789)

    assert torch.allclose(result1, result2), f"Integration with Randomize should be deterministic, max diff: {torch.max(torch.abs(result1 - result2))}"


def test_integration_with_randomize_different_seeds():
    """Test TorchvisionWrapper + Randomize produces different results with different seeds."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 32, 32)

    wrapped_transform = TorchvisionWrapper(T.ColorJitter, brightness=0.5)
    randomize_transform = Randomize(transform=wrapped_transform, p=1.0)

    result1 = randomize_transform(sample_image.clone(), seed=123)
    result2 = randomize_transform(sample_image.clone(), seed=456)

    assert not torch.allclose(result1, result2), "Different seeds should produce different results"


def test_integration_with_randomize_probability():
    """Test TorchvisionWrapper + Randomize respects probability settings."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 32, 32)

    wrapped_transform = TorchvisionWrapper(T.ColorJitter, brightness=0.8)

    # Test with p=0.0 (never apply)
    never_apply = Randomize(transform=wrapped_transform, p=0.0)
    result_never = never_apply(sample_image.clone(), seed=123)

    assert torch.allclose(sample_image, result_never), "p=0.0 should never apply transform"

    # Test with p=1.0 (always apply)
    always_apply = Randomize(transform=wrapped_transform, p=1.0)
    result_always = always_apply(sample_image.clone(), seed=123)

    assert not torch.allclose(sample_image, result_always), "p=1.0 should always apply transform"


def test_multiple_wrapped_transforms_in_sequence():
    """Test multiple TorchvisionWrapper transforms in sequence."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)

    # Create multiple wrapped transforms
    color_jitter = TorchvisionWrapper(T.ColorJitter, brightness=0.3)
    rotation = TorchvisionWrapper(T.RandomRotation, degrees=10)

    # Apply in sequence with generators
    gen1 = torch.Generator()
    gen1.manual_seed(111)
    result1 = color_jitter._call_single(sample_image.clone(), gen1)

    gen2 = torch.Generator()
    gen2.manual_seed(222)
    result2 = rotation._call_single(result1, gen2)

    # Test determinism by repeating
    gen3 = torch.Generator()
    gen3.manual_seed(111)
    result3 = color_jitter._call_single(sample_image.clone(), gen3)

    gen4 = torch.Generator()
    gen4.manual_seed(222)
    result4 = rotation._call_single(result3, gen4)

    assert torch.allclose(result2, result4), "Sequential wrapped transforms should be deterministic"


def test_baseclass_call_method_integration():
    """Test that TorchvisionWrapper works with BaseTransform __call__ method."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 32, 32)

    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.4)

    # Test calling via BaseTransform.__call__ (which uses _call_single_with_generator)
    result1 = wrapper(sample_image.clone(), seed=555)
    result2 = wrapper(sample_image.clone(), seed=555)

    assert torch.allclose(result1, result2), "BaseTransform.__call__ integration should be deterministic"


def test_wrapper_with_complex_transform_args():
    """Test TorchvisionWrapper with complex transform arguments."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 128, 128)

    # Test with RandomAffine with multiple parameters
    wrapper = TorchvisionWrapper(
        T.RandomAffine,
        degrees=(-15, 15),
        translate=(0.1, 0.2),
        scale=(0.9, 1.1),
        shear=(-10, 10)
    )

    gen1 = torch.Generator()
    gen1.manual_seed(777)
    gen2 = torch.Generator()
    gen2.manual_seed(777)

    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)

    assert torch.allclose(result1, result2), "Complex transform args should work deterministically"
    assert result1.shape == sample_image.shape, "Should preserve image dimensions"