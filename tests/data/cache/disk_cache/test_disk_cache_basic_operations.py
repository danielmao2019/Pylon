"""Test basic disk cache put/get operations."""

import pytest
import tempfile
import os
import torch
from data.cache.disk_dataset_cache import DiskDatasetCache
from utils.ops import buffer_equal


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_datapoint():
    """Create a sample datapoint for testing."""
    return {
        'inputs': {
            'image': torch.randn(3, 64, 64),
            'features': torch.randn(256)
        },
        'labels': {
            'mask': torch.randint(0, 2, (64, 64))
        },
        'meta_info': {
            'path': '/test/image.jpg',
            'index': 42,
            'dataset': 'test'
        }
    }


def test_cache_initialization(temp_cache_dir):
    """Test cache initialization and directory structure."""
    version_hash = "test_version_123"
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=True
    )
    
    # Verify cache properties
    assert cache.cache_dir == temp_cache_dir
    assert cache.version_hash == version_hash
    assert cache.enable_validation is True
    
    # Verify directory structure
    version_dir = os.path.join(temp_cache_dir, version_hash)
    assert os.path.exists(version_dir)
    assert cache.version_dir == version_dir


def test_cache_miss(temp_cache_dir):
    """Test cache miss behavior."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="miss_test",
        enable_validation=False
    )
    
    # Test cache miss
    result = cache.get(0)
    assert result is None
    
    # Test exists check
    assert cache.exists(0) is False


def test_put_and_get_basic(temp_cache_dir, sample_datapoint):
    """Test basic put and get operations."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="basic_ops_test",
        enable_validation=False
    )
    
    # Test put operation
    cache.put(0, sample_datapoint)
    
    # Verify cache file exists
    cache_file = cache._get_cache_filepath(0)
    assert os.path.exists(cache_file)
    assert cache.exists(0) is True
    
    # Test get operation
    result = cache.get(0)
    assert result is not None
    assert 'inputs' in result
    assert 'labels' in result
    assert 'meta_info' in result


def test_data_integrity(temp_cache_dir, sample_datapoint):
    """Test that retrieved data matches stored data exactly."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="integrity_test",
        enable_validation=False
    )
    
    # Store and retrieve data
    cache.put(0, sample_datapoint)
    result = cache.get(0)
    
    # Verify data integrity
    assert torch.equal(result['inputs']['image'], sample_datapoint['inputs']['image'])
    assert torch.equal(result['inputs']['features'], sample_datapoint['inputs']['features'])
    assert torch.equal(result['labels']['mask'], sample_datapoint['labels']['mask'])
    assert result['meta_info'] == sample_datapoint['meta_info']


def test_device_transfer_on_get(temp_cache_dir, sample_datapoint):
    """Test device transfer during get operations."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="device_test",
        enable_validation=False
    )
    
    # Store data
    cache.put(0, sample_datapoint)
    
    # Test loading to CPU
    result_cpu = cache.get(0, device='cpu')
    assert result_cpu['inputs']['image'].device.type == 'cpu'
    
    # Test loading to CUDA (if available)
    if torch.cuda.is_available():
        result_cuda = cache.get(0, device='cuda')
        assert result_cuda['inputs']['image'].device.type == 'cuda'


def test_multiple_items(temp_cache_dir, sample_datapoint):
    """Test storing and retrieving multiple items."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="multi_test",
        enable_validation=False
    )
    
    # Store multiple items
    num_items = 10
    for i in range(num_items):
        # Modify the datapoint slightly for each item
        modified_datapoint = {
            'inputs': {
                'image': sample_datapoint['inputs']['image'] + i * 0.1,
                'features': sample_datapoint['inputs']['features'] + i * 0.1
            },
            'labels': sample_datapoint['labels'],
            'meta_info': {**sample_datapoint['meta_info'], 'index': i}
        }
        cache.put(i, modified_datapoint)
    
    # Verify all items can be retrieved
    for i in range(num_items):
        result = cache.get(i)
        assert result is not None
        assert result['meta_info']['index'] == i
        assert cache.exists(i) is True
    
    # Verify cache size
    assert cache.get_size() == num_items


def test_cache_filepath_generation(temp_cache_dir):
    """Test cache file path generation."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="path_test",
        enable_validation=False
    )
    
    # Test various indices
    test_indices = [0, 1, 42, 999, 1000000]
    for idx in test_indices:
        filepath = cache._get_cache_filepath(idx)
        expected_path = os.path.join(cache.version_dir, f"{idx}.pt")
        assert filepath == expected_path


def test_clear_operations(temp_cache_dir, sample_datapoint):
    """Test cache clearing operations."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="clear_test",
        enable_validation=False
    )
    
    # Add multiple items
    for i in range(5):
        cache.put(i, sample_datapoint)
    
    # Verify all files exist
    for i in range(5):
        assert cache.exists(i)
    assert cache.get_size() == 5
    
    # Clear cache
    cache.clear()
    
    # Verify all cache files are removed
    for i in range(5):
        assert not cache.exists(i)
    assert cache.get_size() == 0
    
    # Verify version directory still exists but is empty
    assert os.path.exists(cache.version_dir)
    cache_files = [f for f in os.listdir(cache.version_dir) if f.endswith('.pt')]
    assert len(cache_files) == 0


def test_nonexistent_cache_directory():
    """Test behavior with non-existent cache directory."""
    # Should create directory structure
    with tempfile.TemporaryDirectory() as temp_parent:
        cache_dir = os.path.join(temp_parent, "new_cache")
        cache = DiskDatasetCache(
            cache_dir=cache_dir,
            version_hash="create_test",
            enable_validation=False
        )
        
        assert os.path.exists(cache_dir)
        assert os.path.exists(cache.version_dir)


def test_cache_size_tracking(temp_cache_dir, sample_datapoint):
    """Test that cache correctly tracks its size."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="size_tracking_test",
        enable_validation=False
    )
    
    # Initial size should be 0
    assert cache.get_size() == 0
    
    # Add items and verify size tracking
    for i in range(5):
        cache.put(i, sample_datapoint)
        assert cache.get_size() == i + 1
    
    # Clear cache and verify size resets
    cache.clear()
    assert cache.get_size() == 0


def test_disk_memory_usage_calculation(temp_cache_dir, sample_datapoint):
    """Test disk space usage calculation for cached items."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="memory_usage_test",
        enable_validation=False
    )
    
    # Add item and verify disk usage
    cache.put(0, sample_datapoint)
    
    # Calculate actual file size
    cache_file = cache._get_cache_filepath(0)
    actual_file_size = os.path.getsize(cache_file)
    
    # File should exist and have reasonable size for the test data
    # Image tensor: 3 * 64 * 64 * 4 bytes (float32) = ~49KB
    # Features tensor: 256 * 4 bytes (float32) = ~1KB  
    # Mask tensor: 64 * 64 * 8 bytes (int64) = ~33KB
    # Plus metadata and PyTorch overhead
    expected_min_size = 50 * 1024  # At least 50KB
    expected_max_size = 500 * 1024  # Less than 500KB
    
    assert actual_file_size >= expected_min_size
    assert actual_file_size <= expected_max_size
    
    # Add second item and verify total disk usage increases
    initial_size = actual_file_size
    cache.put(1, sample_datapoint)
    
    cache_file_2 = cache._get_cache_filepath(1)
    total_size = os.path.getsize(cache_file) + os.path.getsize(cache_file_2)
    
    assert total_size > initial_size
    assert cache.get_size() == 2


def test_cache_consistency_with_split_percentages(SampleDataset, temp_cache_dir):
    """Test that cache works consistently with split percentages across multiple instantiations.
    
    This test verifies the fix for the deterministic shuffle issue:
    - Without deterministic shuffle: random.shuffle() causes different splits each time,
      making cached data inconsistent across dataset instances
    - With deterministic shuffle: same base_seed produces same splits, enabling cache reuse
    """
    dataset = SampleDataset(
        data_root=temp_cache_dir,
        split=(0.5, 0.5, 0.0, 0.0),
        base_seed=42,
        use_cpu_cache=False,
        use_disk_cache=True
    )
    # trigger disk cache
    datapoint = dataset[0]
    # re-initialize
    dataset = SampleDataset(
        data_root=temp_cache_dir,
        split=(0.5, 0.5, 0.0, 0.0),
        base_seed=42,
        use_cpu_cache=False,
        use_disk_cache=True
    )
    datapoint = dataset[0]
    expected_datapoint = {
        'inputs': {
            'input': torch.tensor(dataset.annotations[0], dtype=torch.float32, device=dataset.device),
        },
        'labels': {
            'label': torch.tensor(dataset.annotations[0], dtype=torch.float32, device=dataset.device),
        },
        'meta_info': {
            'idx': 0,
        },
    }
    assert buffer_equal(datapoint, expected_datapoint)
