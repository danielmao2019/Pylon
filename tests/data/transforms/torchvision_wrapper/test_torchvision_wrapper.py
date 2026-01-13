"""
Core functionality tests for TorchvisionWrapper.

Tests deterministic behavior and basic wrapper functionality.
"""
import torch
import torchvision.transforms as T
from data.transforms.torchvision_wrapper import TorchvisionWrapper


def test_deterministic_behavior():
    """Test that TorchvisionWrapper produces deterministic results with same seed."""
    # Create sample image tensor (C, H, W) with values in [0, 1]
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)

    # Create wrapper for ColorJitter
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.5, contrast=0.5)

    # Test 1: Same generator seed should produce same results
    gen1 = torch.Generator()
    gen1.manual_seed(123)

    gen2 = torch.Generator()
    gen2.manual_seed(123)

    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)

    assert torch.allclose(result1, result2), f"Same seed should produce identical results, max diff: {torch.max(torch.abs(result1 - result2))}"


def test_random_behavior_with_different_seeds():
    """Test that different seeds produce different results for random transforms."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)

    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.5, contrast=0.5)

    gen1 = torch.Generator()
    gen1.manual_seed(123)

    gen2 = torch.Generator()
    gen2.manual_seed(456)

    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)

    assert not torch.allclose(result1, result2), "Different seeds should produce different results"


def test_multiple_calls_same_seed():
    """Test that multiple calls with same seed are consistent."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 32, 32)

    wrapper = TorchvisionWrapper(T.RandomRotation, degrees=30)

    results = []
    for i in range(3):
        gen = torch.Generator()
        gen.manual_seed(789)  # Same seed each time
        result = wrapper._call_single(sample_image.clone(), gen)
        results.append(result)

    # All results should be identical
    for i in range(1, len(results)):
        assert torch.allclose(results[0], results[i]), f"Multiple calls with same seed should be identical, call {i} differs"


def test_no_generator_fallback():
    """Test that wrapper works without generator (non-deterministic fallback)."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 32, 32)

    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.3)

    # Should not crash with None generator
    result1 = wrapper._call_single(sample_image.clone(), None)
    result2 = wrapper._call_single(sample_image.clone(), None)

    assert isinstance(result1, torch.Tensor), "Should return tensor with None generator"
    assert isinstance(result2, torch.Tensor), "Should return tensor with None generator"
    assert result1.shape == sample_image.shape, "Should preserve image shape"