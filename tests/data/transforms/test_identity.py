"""Test identity transform functionality."""
from typing import List
import pytest
import torch
import numpy as np
from data.transforms.identity import Identity


def test_identity_initialization():
    """Test Identity transform initialization."""
    transform = Identity()
    assert isinstance(transform, Identity)


def test_identity_single_tensor():
    """Test Identity with single tensor input."""
    transform = Identity()
    
    # Test with torch tensor
    tensor = torch.randn(3, 224, 224)
    result = transform(tensor)
    
    # Should return tuple of inputs
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert torch.equal(result[0], tensor)
    assert result[0] is tensor  # Should be same object


def test_identity_multiple_tensors():
    """Test Identity with multiple tensor inputs."""
    transform = Identity()
    
    # Test with multiple tensors
    tensor1 = torch.randn(3, 224, 224)
    tensor2 = torch.randn(1, 224, 224)
    tensor3 = torch.randn(10, 5)
    
    result = transform(tensor1, tensor2, tensor3)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert torch.equal(result[0], tensor1)
    assert torch.equal(result[1], tensor2)
    assert torch.equal(result[2], tensor3)
    
    # Should be same objects
    assert result[0] is tensor1
    assert result[1] is tensor2
    assert result[2] is tensor3


def test_identity_numpy_arrays():
    """Test Identity with numpy array inputs."""
    transform = Identity()
    
    # Test with numpy arrays
    array1 = np.random.randn(3, 224, 224)
    array2 = np.random.randn(10)
    
    result = transform(array1, array2)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert np.array_equal(result[0], array1)
    assert np.array_equal(result[1], array2)
    assert result[0] is array1
    assert result[1] is array2


def test_identity_mixed_types():
    """Test Identity with mixed input types."""
    transform = Identity()
    
    # Test with mixed types
    tensor = torch.randn(3, 3)
    array = np.random.randn(2, 2)
    scalar = 42
    string = "test"
    list_data = [1, 2, 3]
    dict_data = {'key': 'value'}
    
    result = transform(tensor, array, scalar, string, list_data, dict_data)
    
    assert isinstance(result, tuple)
    assert len(result) == 6
    assert torch.equal(result[0], tensor)
    assert np.array_equal(result[1], array)
    assert result[2] == scalar
    assert result[3] == string
    assert result[4] == list_data
    assert result[5] == dict_data
    
    # All should be same objects
    assert all(result[i] is arg for i, arg in enumerate([tensor, array, scalar, string, list_data, dict_data]))


def test_identity_empty_input():
    """Test Identity with no inputs."""
    transform = Identity()
    
    result = transform()
    
    assert isinstance(result, tuple)
    assert len(result) == 0


def test_identity_deterministic():
    """Test Identity transform is deterministic."""
    transform = Identity()
    
    # Create test data
    tensor = torch.randn(5, 5)
    
    # Multiple calls should return same result
    result1 = transform(tensor)
    result2 = transform(tensor)
    
    assert torch.equal(result1[0], result2[0])
    assert result1[0] is tensor
    assert result2[0] is tensor


@pytest.mark.parametrize("input_shape", [
    (1,),
    (10, 10),
    (3, 224, 224),
    (2, 3, 224, 224),
    (1, 1, 1, 1, 1),
])
def test_identity_various_shapes(input_shape):
    """Test Identity with various tensor shapes."""
    transform = Identity()
    
    tensor = torch.randn(*input_shape)
    result = transform(tensor)
    
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert torch.equal(result[0], tensor)
    assert result[0].shape == input_shape


def test_identity_with_requires_grad():
    """Test Identity preserves gradient requirements."""
    transform = Identity()
    
    # Test with gradient-enabled tensor
    tensor = torch.randn(3, 3, requires_grad=True)
    result = transform(tensor)
    
    assert result[0].requires_grad == True
    assert result[0] is tensor
    
    # Test with gradient-disabled tensor
    tensor_no_grad = torch.randn(3, 3, requires_grad=False)
    result_no_grad = transform(tensor_no_grad)
    
    assert result_no_grad[0].requires_grad == False
    assert result_no_grad[0] is tensor_no_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_identity_gpu_tensors():
    """Test Identity with GPU tensors."""
    transform = Identity()
    
    # Test with GPU tensor
    tensor_gpu = torch.randn(3, 3, device='cuda')
    result = transform(tensor_gpu)
    
    assert result[0].is_cuda
    assert torch.equal(result[0], tensor_gpu)
    assert result[0] is tensor_gpu