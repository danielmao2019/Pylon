"""
Tests for TorchvisionWrapper with multiple different torchvision transforms.

Tests various torchvision transforms to ensure the wrapper works generically.
"""
import torch
import torchvision.transforms as T
from data.transforms.torchvision_wrapper import TorchvisionWrapper


def test_color_jitter_wrapper():
    """Test TorchvisionWrapper with ColorJitter."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)
    
    wrapper = TorchvisionWrapper(T.ColorJitter, brightness=0.5, contrast=0.5)
    
    gen1 = torch.Generator()
    gen1.manual_seed(123)
    gen2 = torch.Generator()
    gen2.manual_seed(123)
    
    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)
    
    assert torch.allclose(result1, result2), "ColorJitter wrapper should be deterministic"


def test_random_affine_wrapper():
    """Test TorchvisionWrapper with RandomAffine."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)
    
    wrapper = TorchvisionWrapper(T.RandomAffine, degrees=10, translate=(0.1, 0.1))
    
    gen1 = torch.Generator()
    gen1.manual_seed(456)
    gen2 = torch.Generator()
    gen2.manual_seed(456)
    
    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)
    
    assert torch.allclose(result1, result2), "RandomAffine wrapper should be deterministic"


def test_random_rotation_wrapper():
    """Test TorchvisionWrapper with RandomRotation."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)
    
    wrapper = TorchvisionWrapper(T.RandomRotation, degrees=30)
    
    gen1 = torch.Generator()
    gen1.manual_seed(789)
    gen2 = torch.Generator()
    gen2.manual_seed(789)
    
    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)
    
    assert torch.allclose(result1, result2), "RandomRotation wrapper should be deterministic"


def test_random_horizontal_flip_wrapper():
    """Test TorchvisionWrapper with RandomHorizontalFlip."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)
    
    wrapper = TorchvisionWrapper(T.RandomHorizontalFlip, p=0.5)
    
    gen1 = torch.Generator()
    gen1.manual_seed(999)
    gen2 = torch.Generator()
    gen2.manual_seed(999)
    
    result1 = wrapper._call_single(sample_image.clone(), gen1)
    result2 = wrapper._call_single(sample_image.clone(), gen2)
    
    assert torch.allclose(result1, result2), "RandomHorizontalFlip wrapper should be deterministic"


def test_different_transforms_different_results():
    """Test that different transforms produce different results."""
    torch.manual_seed(42)
    sample_image = torch.rand(3, 64, 64)
    
    wrapper1 = TorchvisionWrapper(T.ColorJitter, brightness=0.5)
    wrapper2 = TorchvisionWrapper(T.RandomRotation, degrees=10)
    
    gen1 = torch.Generator()
    gen1.manual_seed(123)
    gen2 = torch.Generator()
    gen2.manual_seed(123)
    
    result1 = wrapper1._call_single(sample_image.clone(), gen1)
    result2 = wrapper2._call_single(sample_image.clone(), gen2)
    
    # Different transforms should produce different results even with same seed
    assert not torch.allclose(result1, result2), "Different transforms should produce different results"