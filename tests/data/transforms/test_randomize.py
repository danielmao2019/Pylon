"""Test randomize transform functionality."""
from typing import List, Union
import pytest
import torch
import numpy as np
from data.transforms.randomize import Randomize
from data.transforms.base_transform import BaseTransform


class MockTransform:
    """Mock transform for testing."""
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, *args) -> Union[torch.Tensor, List[torch.Tensor]]:
        self.call_count += 1
        # Add 1 to all inputs
        if len(args) == 1:
            return args[0] + 1
        else:
            return [arg + 1 for arg in args]


def test_randomize_initialization():
    """Test Randomize transform initialization."""
    mock_transform = MockTransform()
    
    # Valid initialization
    transform = Randomize(transform=mock_transform, p=0.5)
    assert transform.p == 0.5
    assert transform.transform is mock_transform
    
    # Test with p=0 and p=1
    transform_never = Randomize(transform=mock_transform, p=0.0)
    assert transform_never.p == 0.0
    
    transform_always = Randomize(transform=mock_transform, p=1.0)
    assert transform_always.p == 1.0


def test_randomize_invalid_initialization():
    """Test Randomize transform with invalid initialization."""
    mock_transform = MockTransform()
    
    # Invalid probability
    with pytest.raises(AssertionError):
        Randomize(transform=mock_transform, p=1.5)
    
    with pytest.raises(AssertionError):
        Randomize(transform=mock_transform, p=-0.1)
    
    # Invalid transform type
    with pytest.raises(AssertionError):
        Randomize(transform="not_callable", p=0.5)
    
    # Cannot use BaseTransform or Compose directly
    # Note: This check was removed in the implementation


def test_randomize_always_apply():
    """Test Randomize with p=1 (always apply)."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=1.0)
    
    # Single tensor
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = transform(tensor)
    
    assert torch.equal(result, tensor + 1)
    assert mock_transform.call_count == 1
    
    # Multiple calls should always apply
    for i in range(10):
        result = transform(tensor)
        assert torch.equal(result, tensor + 1)
    
    assert mock_transform.call_count == 11


def test_randomize_never_apply():
    """Test Randomize with p=0 (never apply)."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=0.0)
    
    # Single tensor
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = transform(tensor)
    
    assert torch.equal(result, tensor)
    assert mock_transform.call_count == 0
    
    # Multiple calls should never apply
    for i in range(10):
        result = transform(tensor)
        assert torch.equal(result, tensor)
    
    assert mock_transform.call_count == 0


def test_randomize_deterministic_with_seed():
    """Test Randomize is deterministic with same seed."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=0.5)
    
    tensor = torch.tensor([1.0, 2.0, 3.0])
    
    # First run with seed
    results1 = []
    for i in range(10):
        result = transform(tensor, seed=42 + i)
        results1.append(torch.equal(result, tensor + 1))
    
    # Reset call count
    mock_transform.call_count = 0
    
    # Second run with same seeds
    results2 = []
    for i in range(10):
        result = transform(tensor, seed=42 + i)
        results2.append(torch.equal(result, tensor + 1))
    
    # Results should be identical
    assert results1 == results2


def test_randomize_single_input():
    """Test Randomize with single input."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=1.0)
    
    tensor = torch.randn(3, 3)
    result = transform(tensor)
    
    # Should return single tensor, not list
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, tensor + 1)


def test_randomize_multiple_inputs():
    """Test Randomize with multiple inputs."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=1.0)
    
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(2, 2)
    
    result = transform(tensor1, tensor2)
    
    # Should return list of tensors
    assert isinstance(result, list)
    assert len(result) == 2
    assert torch.equal(result[0], tensor1 + 1)
    assert torch.equal(result[1], tensor2 + 1)


def test_randomize_no_application_returns_correct_type():
    """Test Randomize returns correct type when transform is not applied."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=0.0)
    
    # Single input should return tensor
    tensor = torch.randn(3, 3)
    result = transform(tensor)
    assert isinstance(result, torch.Tensor)
    assert result is tensor
    
    # Multiple inputs should return list
    tensor1 = torch.randn(3, 3)
    tensor2 = torch.randn(2, 2)
    result = transform(tensor1, tensor2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] is tensor1
    assert result[1] is tensor2


def test_randomize_probability_distribution():
    """Test Randomize applies transform with correct probability."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=0.3)
    
    # Run many times and check application rate
    num_trials = 1000
    applications = 0
    
    tensor = torch.tensor([1.0])
    for i in range(num_trials):
        result = transform(tensor, seed=i)
        if torch.equal(result, tensor + 1):
            applications += 1
    
    # Should be approximately 30%
    application_rate = applications / num_trials
    assert 0.25 < application_rate < 0.35  # Allow some variance


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_randomize_various_probabilities(p):
    """Test Randomize with various probability values."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=p)
    
    tensor = torch.randn(5)
    
    # Just ensure it runs without error
    for i in range(10):
        result = transform(tensor, seed=i)
        assert isinstance(result, torch.Tensor)
        
        # Check if transform was applied
        if torch.equal(result, tensor + 1):
            assert mock_transform.call_count > 0


def test_randomize_with_numpy_inputs():
    """Test Randomize with numpy array inputs."""
    mock_transform = MockTransform()
    transform = Randomize(transform=mock_transform, p=1.0)
    
    array = np.array([1.0, 2.0, 3.0])
    result = transform(array)
    
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, array + 1)