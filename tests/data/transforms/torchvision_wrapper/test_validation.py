"""
Validation tests for TorchvisionWrapper.

Tests that valid inputs are accepted and processed correctly.
"""
import torch
import torchvision.transforms as T
from data.transforms.torchvision_wrapper import TorchvisionWrapper


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


def test_valid_transform_parameters():
    """Test that various valid transform parameters are accepted."""
    # Test ColorJitter with various parameter combinations
    wrapper1 = TorchvisionWrapper(T.ColorJitter, brightness=0.5)
    wrapper2 = TorchvisionWrapper(T.ColorJitter, brightness=0.2, contrast=0.3)
    wrapper3 = TorchvisionWrapper(T.ColorJitter, brightness=(0.8, 1.2), contrast=(0.8, 1.2))

    # Test RandomAffine with complex parameters
    wrapper4 = TorchvisionWrapper(
        T.RandomAffine,
        degrees=(-15, 15),
        translate=(0.1, 0.2),
        scale=(0.9, 1.1),
        shear=(-10, 10)
    )

    # Test RandomRotation with different parameter types
    wrapper5 = TorchvisionWrapper(T.RandomRotation, degrees=30)
    wrapper6 = TorchvisionWrapper(T.RandomRotation, degrees=(-45, 45))

    # All should initialize successfully
    sample_image = torch.rand(3, 32, 32)
    for wrapper in [wrapper1, wrapper2, wrapper3, wrapper4, wrapper5, wrapper6]:
        result = wrapper._call_single(sample_image.clone(), None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_image.shape


def test_valid_generator_input():
    """Test that valid generator inputs are accepted."""
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.4)
    sample_image = torch.rand(3, 32, 32)

    # Test with different generator devices
    generators = [
        torch.Generator(device='cpu'),
        torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu'),
    ]

    for gen in generators:
        gen.manual_seed(42)
        result = wrapper._call_single(sample_image.clone(), gen)
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_image.shape


def test_valid_seed_values():
    """Test that various valid seed values work correctly."""
    wrapper = TorchvisionWrapper(T.RandomRotation, degrees=15)
    sample_image = torch.rand(3, 32, 32)

    # Test with different seed values
    valid_seeds = [0, 1, 42, 999, 123456, 2**31 - 1]

    for seed in valid_seeds:
        gen = torch.Generator()
        gen.manual_seed(seed)
        result = wrapper._call_single(sample_image.clone(), gen)
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_image.shape