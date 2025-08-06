import pytest
import torch
import tempfile
import copy
from data.cache.disk_dataset_cache import DiskDatasetCache


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
def disk_cache():
    """Create a temporary disk cache for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DiskDatasetCache(
            cache_dir=temp_dir,
            version_hash='test_version',
            enable_validation=True,
        )
        yield cache


@pytest.fixture
def disk_cache_with_items(disk_cache, sample_datapoint):
    """Create a disk cache pre-populated with 3 items."""
    for i in range(3):
        disk_cache.put(i, copy.deepcopy(sample_datapoint))
    return disk_cache


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
