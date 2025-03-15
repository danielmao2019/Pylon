import pytest
import torch
import psutil
from data.cache import DatasetCache


@pytest.fixture
def tensor_params():
    """Parameters for creating test tensors."""
    return {
        'dim': 1024,
        'channels': 3,
        'dtype': torch.float32
    }


@pytest.fixture
def sample_tensor(tensor_params):
    """Create a sample tensor for testing."""
    return torch.randn(
        tensor_params['channels'],
        tensor_params['dim'],
        tensor_params['dim'],
        dtype=tensor_params['dtype']
    )


@pytest.fixture
def sample_datapoint(sample_tensor):
    """Create a sample datapoint with tensor, label, and metadata."""
    return {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([0])},
        'meta_info': {'filename': 'test.jpg'}
    }


@pytest.fixture
def three_item_cache(sample_datapoint):
    """Create a cache configured to hold exactly 3 items."""
    # Calculate memory needed for 3 items
    item_memory = DatasetCache._calculate_item_memory(sample_datapoint)
    total_memory_needed = item_memory * 3
    
    # Set percentage to allow exactly 3 items
    max_percent = (total_memory_needed / psutil.virtual_memory().total) * 100
    
    return DatasetCache(max_memory_percent=max_percent)


@pytest.fixture
def cache_with_items(three_item_cache, sample_datapoint):
    """Create a cache pre-populated with 3 items."""
    cache = three_item_cache
    for i in range(3):
        cache.put(i, sample_datapoint)
    return cache


@pytest.fixture
def make_datapoint(tensor_params):
    """Factory fixture to create datapoints with unique tensors and metadata."""
    def _make_datapoint(index: int):
        tensor = torch.randn(
            tensor_params['channels'],
            tensor_params['dim'],
            tensor_params['dim'],
            dtype=tensor_params['dtype']
        )
        return {
            'inputs': {'image': tensor},
            'labels': {'class': torch.tensor([index])},
            'meta_info': {'filename': f'test_{index}.jpg'}
        }
    return _make_datapoint
