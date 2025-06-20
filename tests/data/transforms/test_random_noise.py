"""Test random noise transform functionality."""
import pytest
import torch
import numpy as np
from data.transforms.random_noise import RandomNoise


# Skip all RandomNoise tests if CUDA is not available
# This is because BaseTransform._get_generator creates CUDA generators
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="RandomNoise requires CUDA due to BaseTransform generator")


@pytest.fixture
def device():
    """Get the device to use for testing (CUDA required)."""
    return torch.device('cuda')


def test_random_noise_initialization():
    """Test RandomNoise transform initialization."""
    # Default initialization
    transform = RandomNoise()
    assert transform.std == 0.1
    
    # Custom std
    transform = RandomNoise(std=0.5)
    assert transform.std == 0.5
    
    # Zero std
    transform = RandomNoise(std=0.0)
    assert transform.std == 0.0


def test_random_noise_single_tensor(device):
    """Test RandomNoise with single tensor input."""
    transform = RandomNoise(std=0.1)
    
    # Create input tensor
    tensor = torch.ones(3, 224, 224, device=device)
    
    # Apply transform
    result = transform(tensor)
    
    # Check output properties
    assert isinstance(result, torch.Tensor)
    assert result.shape == tensor.shape
    assert result.dtype == tensor.dtype
    assert result.device == tensor.device
    
    # Should be different from input (unless std=0)
    assert not torch.equal(result, tensor)
    
    # Check noise statistics
    noise = result - tensor
    assert torch.abs(noise.mean()) < 0.01  # Mean should be close to 0
    assert torch.abs(noise.std() - 0.1) < 0.01  # Std should be close to 0.1


def test_random_noise_zero_std(device):
    """Test RandomNoise with zero standard deviation."""
    transform = RandomNoise(std=0.0)
    
    tensor = torch.randn(5, 5, device=device)
    result = transform(tensor)
    
    # Should be identical to input
    assert torch.equal(result, tensor)


def test_random_noise_deterministic_with_seed(device):
    """Test RandomNoise is deterministic with same seed."""
    transform = RandomNoise(std=0.2)
    
    tensor = torch.zeros(10, 10, device=device)
    
    # First run with seed
    result1 = transform(tensor, seed=42)
    
    # Second run with same seed
    result2 = transform(tensor, seed=42)
    
    # Should be identical
    assert torch.equal(result1, result2)
    
    # Third run with different seed
    result3 = transform(tensor, seed=43)
    
    # Should be different
    assert not torch.equal(result1, result3)


def test_random_noise_preserves_dtype(device):
    """Test RandomNoise preserves tensor dtype."""
    transform = RandomNoise(std=0.1)
    
    # Test different dtypes
    dtypes = [torch.float32, torch.float64, torch.float16]
    
    for dtype in dtypes:
        tensor = torch.ones(3, 3, dtype=dtype, device=device)
        result = transform(tensor)
        assert result.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_noise_gpu_tensors():
    """Test RandomNoise with GPU tensors."""
    transform = RandomNoise(std=0.15)
    
    # Create GPU tensor
    tensor_gpu = torch.ones(5, 5, device='cuda')
    
    # Apply transform
    result = transform(tensor_gpu)
    
    # Check output is on GPU
    assert result.is_cuda
    assert result.device == tensor_gpu.device
    
    # Check noise was added
    assert not torch.equal(result, tensor_gpu)
    
    # Check noise statistics (relax tolerance for statistical variance)
    noise = result - tensor_gpu
    assert torch.abs(noise.std() - 0.15) < 0.05


def test_random_noise_various_shapes(device):
    """Test RandomNoise with various tensor shapes."""
    transform = RandomNoise(std=0.1)
    
    shapes = [
        (10,),           # 1D
        (5, 5),          # 2D
        (3, 224, 224),   # 3D (image-like)
        (2, 3, 224, 224), # 4D (batch)
        (1, 1, 1, 1, 1)  # 5D
    ]
    
    for shape in shapes:
        tensor = torch.zeros(shape, device=device)
        result = transform(tensor)
        
        assert result.shape == shape
        # All values should be approximately distributed around 0 with std 0.1
        # Relax tolerance for small tensors where statistical variance is high
        mean_tolerance = 0.2 if torch.numel(tensor) < 10 else 0.05
        assert torch.abs(result.mean()) < mean_tolerance, f"{result.mean()=}, {shape=}"
        
        # Only check std for tensors with more than 1 element (std is NaN for single element)
        if torch.numel(tensor) > 1:
            std_tolerance = 0.1 if torch.numel(tensor) < 10 else 0.05
            assert torch.abs(result.std() - 0.1) < std_tolerance


def test_random_noise_large_std(device):
    """Test RandomNoise with large standard deviation."""
    transform = RandomNoise(std=10.0)
    
    tensor = torch.zeros(1000, 1000, device=device)
    result = transform(tensor)
    
    # Check noise statistics with larger sample
    assert torch.abs(result.mean()) < 0.5  # Mean should still be close to 0
    assert torch.abs(result.std() - 10.0) < 0.5  # Std should be close to 10


def test_random_noise_maintains_gradient(device):
    """Test RandomNoise maintains gradient computation."""
    transform = RandomNoise(std=0.1)
    
    # Create tensor with gradient
    tensor = torch.ones(3, 3, requires_grad=True, device=device)
    
    # Apply transform
    result = transform(tensor)
    
    # Result should require gradient
    assert result.requires_grad
    
    # Gradient should flow back
    loss = result.sum()
    loss.backward()
    
    assert tensor.grad is not None
    assert torch.all(tensor.grad == 1.0)  # Gradient of sum is 1 everywhere


def test_random_noise_batch_consistency(device):
    """Test RandomNoise applies different noise to different samples in batch."""
    transform = RandomNoise(std=0.2)
    
    # Create batch of identical tensors
    batch_size = 4
    tensor = torch.ones(batch_size, 3, 32, 32, device=device)
    
    # Apply transform with same seed
    result = transform(tensor, seed=42)
    
    # Each sample in batch should have different noise
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            assert not torch.equal(result[i], result[j])
    
    # But noise statistics should be consistent across batch
    noise = result - tensor
    for i in range(batch_size):
        sample_noise = noise[i]
        assert torch.abs(sample_noise.mean()) < 0.05
        assert torch.abs(sample_noise.std() - 0.2) < 0.05


@pytest.mark.parametrize("std", [0.01, 0.1, 0.5, 1.0, 5.0])
def test_random_noise_various_stds(std, device):
    """Test RandomNoise with various standard deviations."""
    transform = RandomNoise(std=std)
    
    # Use larger tensor for better statistics
    tensor = torch.zeros(100, 100, device=device)
    result = transform(tensor)
    
    # Check noise statistics
    assert torch.abs(result.mean()) < std * 0.1  # Mean should be small relative to std
    assert torch.abs(result.std() - std) < std * 0.1  # Std should match parameter