import pytest
import torch


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 4, 4)


@pytest.fixture
def sample_tensors():
    """Create multiple sample tensors for testing."""
    return [torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)]


@pytest.fixture
def sample_tensor_dict():
    """Create a dictionary of sample tensors for testing."""
    return {
        'pred1': torch.randn(2, 3, 4, 4),
        'pred2': torch.randn(2, 3, 4, 4)
    }


@pytest.fixture
def sample_multi_task_tensors():
    """Create sample tensors for multi-task testing."""
    return {
        'task1': torch.randn(2, 3, 4, 4),
        'task2': torch.randn(2, 3, 4, 4)
    }
