import pytest
import torch


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(3, 1024, 1024)


@pytest.fixture
def sample_datapoint(sample_tensor):
    """Create a sample datapoint structure for testing."""
    return {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([1])},
        'meta_info': {'filename': 'test.jpg'}
    }
