"""Test disk cache file system operations and atomic writes."""

import pytest
import tempfile
import os
import torch
from unittest.mock import patch
from data.cache.disk_dataset_cache import DiskDatasetCache


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


def test_atomic_write_operations(temp_cache_dir, sample_datapoint):
    """Test atomic write operations using temporary files - fails fast and loud."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="atomic_test",
        enable_validation=False
    )
    
    # Mock os.rename to simulate failure during atomic write
    original_rename = os.rename
    
    def failing_rename(src, dst):
        if dst.endswith('.pt'):  # Only fail for the final rename
            raise OSError("Simulated rename failure")
        return original_rename(src, dst)
    
    with patch('os.rename', side_effect=failing_rename):
        with pytest.raises(RuntimeError, match="Error saving torch file.*Simulated rename failure"):
            cache.put(0, sample_datapoint)
    
    # Verify no partial files remain
    cache_file = cache._get_cache_filepath(0)
    temp_file = cache_file + '.tmp'
    assert not os.path.exists(cache_file)
    assert not os.path.exists(temp_file)


def test_atomic_write_cleanup_on_exception(temp_cache_dir, sample_datapoint):
    """Test that temporary files are cleaned up when exceptions occur."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="cleanup_test",
        enable_validation=False
    )
    
    # Mock torch.save to simulate failure during save
    with patch('torch.save', side_effect=RuntimeError("Save failed")):
        with pytest.raises(RuntimeError, match="Save failed"):
            cache.put(0, sample_datapoint)
    
    # Verify no temporary files remain
    cache_file = cache._get_cache_filepath(0)
    temp_file = cache_file + '.tmp'
    assert not os.path.exists(cache_file)
    assert not os.path.exists(temp_file)


def test_disk_space_usage_tracking(temp_cache_dir, sample_datapoint):
    """Test disk space usage tracking."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="space_test",
        enable_validation=True
    )
    
    # Add item and measure disk usage
    cache.put(0, sample_datapoint)
    
    cache_file = cache._get_cache_filepath(0)
    file_size = os.path.getsize(cache_file)
    
    # File should exist and have reasonable size
    assert file_size > 1000  # Should be at least 1KB
    assert file_size < 50 * 1024 * 1024  # Should be less than 50MB for test data


def test_multiple_files_disk_usage(temp_cache_dir, sample_datapoint):
    """Test disk usage with multiple cache files."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="multi_space_test",
        enable_validation=True
    )
    
    # Test multiple files
    total_size_before = sum(
        os.path.getsize(os.path.join(cache.version_dir, f))
        for f in os.listdir(cache.version_dir)
        if f.endswith('.pt')
    ) if os.path.exists(cache.version_dir) else 0
    
    # Add multiple items
    for i in range(5):
        cache.put(i, sample_datapoint)
    
    total_size_after = sum(
        os.path.getsize(os.path.join(cache.version_dir, f))
        for f in os.listdir(cache.version_dir)
        if f.endswith('.pt')
    )
    
    assert total_size_after > total_size_before
    assert cache.get_size() == 5


def test_file_corruption_during_read(temp_cache_dir, sample_datapoint):
    """Test handling of file corruption during read operations - fails fast and loud."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="read_corruption_test",
        enable_validation=False
    )
    
    # Store valid item
    cache.put(0, sample_datapoint)
    cache_file = cache._get_cache_filepath(0)
    
    # Corrupt file with partial torch data
    with open(cache_file, 'wb') as f:
        f.write(b'\x80\x02}q\x00X\x06\x00\x00\x00')  # Partial pickle data
    
    # Should fail fast and loud - RuntimeError raised immediately
    with pytest.raises(RuntimeError, match="Error loading torch file.*pickle data was truncated"):
        cache.get(0)
    
    # File should still exist - we don't mask errors by removing files
    assert os.path.exists(cache_file)


def test_concurrent_file_access_safety(temp_cache_dir, sample_datapoint):
    """Test file access safety with concurrent operations."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="concurrent_file_test",
        enable_validation=False
    )
    
    import threading
    import time
    
    results = {}
    
    def writer_worker():
        """Worker that writes to cache."""
        try:
            for i in range(10):
                cache.put(i, sample_datapoint)
                time.sleep(0.001)  # Small delay
            results['writer'] = 'success'
        except Exception as e:
            results['writer'] = f'error: {e}'
    
    def reader_worker():
        """Worker that reads from cache."""
        try:
            successful_reads = 0
            for i in range(20):
                result = cache.get(i % 10)
                if result is not None:
                    successful_reads += 1
                time.sleep(0.001)  # Small delay
            results['reader'] = successful_reads
        except Exception as e:
            results['reader'] = f'error: {e}'
    
    # Start concurrent threads
    writer_thread = threading.Thread(target=writer_worker)
    reader_thread = threading.Thread(target=reader_worker)
    
    writer_thread.start()
    reader_thread.start()
    
    writer_thread.join()
    reader_thread.join()
    
    # Verify no errors occurred
    assert results['writer'] == 'success'
    assert isinstance(results['reader'], int)
    assert results['reader'] >= 0


def test_directory_creation_permissions(temp_cache_dir):
    """Test that cache directories are created with proper permissions."""
    version_hash = "permissions_test"
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash=version_hash,
        enable_validation=False
    )
    
    # Check version directory permissions
    version_dir = cache.version_dir
    assert os.path.exists(version_dir)
    assert os.access(version_dir, os.R_OK | os.W_OK | os.X_OK)
    
    # Verify directory has reasonable permissions
    stat_info = os.stat(version_dir)
    # Should be readable/writable/executable by owner at minimum
    assert stat_info.st_mode & 0o700 == 0o700


def test_file_loading_with_different_map_locations(temp_cache_dir, sample_datapoint):
    """Test file loading with different torch map_location parameters."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="map_location_test",
        enable_validation=False
    )
    
    # Store data
    cache.put(0, sample_datapoint)
    
    # Test loading with different map locations
    result_cpu = cache.get(0, device='cpu')
    assert result_cpu is not None
    assert result_cpu['inputs']['image'].device.type == 'cpu'
    
    # Test with explicit None device
    result_default = cache.get(0, device=None)
    assert result_default is not None
    assert result_default['inputs']['image'].device.type == 'cpu'
    
    # Test with string device specification
    if torch.cuda.is_available():
        result_cuda = cache.get(0, device='cuda:0')
        assert result_cuda is not None
        assert result_cuda['inputs']['image'].device.type == 'cuda'


def test_cache_file_format_validation(temp_cache_dir, sample_datapoint):
    """Test that cache files have the expected format."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="format_test",
        enable_validation=True
    )
    
    # Store data
    cache.put(0, sample_datapoint)
    
    # Load raw cache file and verify structure
    cache_file = cache._get_cache_filepath(0)
    cached_data = torch.load(cache_file)
    
    # Verify required keys
    assert 'inputs' in cached_data
    assert 'labels' in cached_data
    assert 'meta_info' in cached_data
    assert 'checksum' in cached_data  # Should have checksum when validation enabled
    
    # Verify data types
    assert isinstance(cached_data['inputs'], dict)
    assert isinstance(cached_data['labels'], dict)
    assert isinstance(cached_data['meta_info'], dict)
    assert isinstance(cached_data['checksum'], str)