import pytest
import torch


@pytest.fixture(scope="session")
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(3, 64, 64)


@pytest.fixture(scope="session")
def sample_datapoint(sample_tensor):
    """Create a sample datapoint structure that mimics real dataset items."""
    return {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([1])},
        'meta_info': {'filename': 'test.jpg'}
    }
