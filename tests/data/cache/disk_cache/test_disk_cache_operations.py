"""Comprehensive tests for disk cache operations.

This module tests disk cache functionality including:
- Basic put/get operations
- File system operations
- Atomic writes and error handling
- Cache validation and corruption detection
- Concurrent access patterns
"""

import pytest
import tempfile
import os
import time
import threading
import torch
from unittest.mock import patch, mock_open

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


class TestDiskCacheBasicOperations:
    """Test basic disk cache operations."""
    
    def test_cache_initialization(self, temp_cache_dir):
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
        
        # Verify metadata file
        metadata_file = os.path.join(temp_cache_dir, 'cache_metadata.json')
        assert os.path.exists(metadata_file)
        assert cache.metadata_file == metadata_file
    
    def test_put_and_get_operations(self, temp_cache_dir, sample_datapoint):
        """Test basic put and get operations."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="basic_ops_test",
            enable_validation=False
        )
        
        # Test cache miss
        result = cache.get(0)
        assert result is None
        
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
        
        # Verify data integrity
        assert torch.equal(result['inputs']['image'], sample_datapoint['inputs']['image'])
        assert torch.equal(result['inputs']['features'], sample_datapoint['inputs']['features'])
        assert torch.equal(result['labels']['mask'], sample_datapoint['labels']['mask'])
        assert result['meta_info'] == sample_datapoint['meta_info']
    
    def test_device_transfer_on_get(self, temp_cache_dir, sample_datapoint):
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
    
    def test_multiple_items(self, temp_cache_dir, sample_datapoint):
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


class TestDiskCacheValidation:
    """Test disk cache validation and corruption detection."""
    
    def test_checksum_validation_enabled(self, temp_cache_dir, sample_datapoint):
        """Test checksum validation when enabled."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="validation_test",
            enable_validation=True
        )
        
        # Store item with validation
        cache.put(0, sample_datapoint)
        
        # Retrieve should succeed with valid checksum
        result = cache.get(0)
        assert result is not None
        
        # Load cached file and verify checksum is stored
        cache_file = cache._get_cache_filepath(0)
        cached_data = torch.load(cache_file)
        assert 'checksum' in cached_data
        assert len(cached_data['checksum']) > 0
    
    def test_checksum_validation_disabled(self, temp_cache_dir, sample_datapoint):
        """Test behavior when validation is disabled."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="no_validation_test",
            enable_validation=False
        )
        
        # Store item without validation
        cache.put(0, sample_datapoint)
        
        # Load cached file and verify no checksum
        cache_file = cache._get_cache_filepath(0)
        cached_data = torch.load(cache_file)
        assert 'checksum' not in cached_data
        
        # Retrieve should work without validation
        result = cache.get(0)
        assert result is not None
    
    def test_corruption_detection(self, temp_cache_dir, sample_datapoint):
        """Test detection and handling of corrupted cache files."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="corruption_test",
            enable_validation=True
        )
        
        # Store valid item
        cache.put(0, sample_datapoint)
        cache_file = cache._get_cache_filepath(0)
        assert os.path.exists(cache_file)
        
        # Corrupt the file by writing invalid data
        with open(cache_file, 'w') as f:
            f.write("This is not valid torch data")
        
        # Attempt to retrieve should return None and remove corrupted file
        result = cache.get(0)
        assert result is None
        assert not os.path.exists(cache_file)
    
    def test_validation_session_tracking(self, temp_cache_dir, sample_datapoint):
        """Test that validation only happens once per session."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="session_test",
            enable_validation=True
        )
        
        # Store item
        cache.put(0, sample_datapoint)
        
        # First access should validate
        assert 0 not in cache.validated_keys
        result1 = cache.get(0)
        assert result1 is not None
        assert 0 in cache.validated_keys
        
        # Mock the validation method to track calls
        with patch.object(cache, '_validate_item', wraps=cache._validate_item) as mock_validate:
            # Second access should skip validation
            result2 = cache.get(0)
            assert result2 is not None
            mock_validate.assert_called_once()  # Should be called but return early


class TestDiskCacheFileOperations:
    """Test disk cache file system operations."""
    
    def test_atomic_write_operations(self, temp_cache_dir, sample_datapoint):
        """Test atomic write operations using temporary files."""
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
            with pytest.raises(OSError, match="Simulated rename failure"):
                cache.put(0, sample_datapoint)
        
        # Verify no partial files remain
        cache_file = cache._get_cache_filepath(0)
        temp_file = cache_file + '.tmp'
        assert not os.path.exists(cache_file)
        assert not os.path.exists(temp_file)
    
    def test_cache_filepath_generation(self, temp_cache_dir):
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
    
    def test_clear_operations(self, temp_cache_dir, sample_datapoint):
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
    
    def test_nonexistent_cache_directory(self):
        """Test behavior with non-existent cache directory."""
        nonexistent_dir = "/nonexistent/cache/dir"
        
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


class TestDiskCacheConcurrency:
    """Test disk cache concurrent access patterns."""
    
    def test_concurrent_put_operations(self, temp_cache_dir, sample_datapoint):
        """Test concurrent put operations with thread safety."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="concurrent_put_test",
            enable_validation=False
        )
        
        num_threads = 10
        items_per_thread = 5
        results = {}
        
        def put_worker(thread_id):
            """Worker function for concurrent puts."""
            try:
                for i in range(items_per_thread):
                    idx = thread_id * items_per_thread + i
                    modified_datapoint = {
                        'inputs': sample_datapoint['inputs'],
                        'labels': sample_datapoint['labels'],
                        'meta_info': {**sample_datapoint['meta_info'], 'thread_id': thread_id, 'item_id': i}
                    }
                    cache.put(idx, modified_datapoint)
                results[thread_id] = 'success'
            except Exception as e:
                results[thread_id] = f'error: {e}'
        
        # Start concurrent threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=put_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        for thread_id in range(num_threads):
            assert results[thread_id] == 'success'
        
        # Verify all items were stored correctly
        total_items = num_threads * items_per_thread
        assert cache.get_size() == total_items
        
        for thread_id in range(num_threads):
            for i in range(items_per_thread):
                idx = thread_id * items_per_thread + i
                result = cache.get(idx)
                assert result is not None
                assert result['meta_info']['thread_id'] == thread_id
                assert result['meta_info']['item_id'] == i
    
    def test_concurrent_get_operations(self, temp_cache_dir, sample_datapoint):
        """Test concurrent get operations."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="concurrent_get_test",
            enable_validation=False
        )
        
        # Pre-populate cache
        num_items = 20
        for i in range(num_items):
            modified_datapoint = {
                'inputs': sample_datapoint['inputs'],
                'labels': sample_datapoint['labels'],
                'meta_info': {**sample_datapoint['meta_info'], 'index': i}
            }
            cache.put(i, modified_datapoint)
        
        # Concurrent read test
        num_threads = 10
        reads_per_thread = 50
        results = {}
        
        def get_worker(thread_id):
            """Worker function for concurrent gets."""
            try:
                successful_reads = 0
                for _ in range(reads_per_thread):
                    idx = thread_id % num_items  # Round-robin through items
                    result = cache.get(idx)
                    if result is not None:
                        successful_reads += 1
                results[thread_id] = successful_reads
            except Exception as e:
                results[thread_id] = f'error: {e}'
        
        # Start concurrent threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=get_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all operations succeeded
        for thread_id in range(num_threads):
            assert isinstance(results[thread_id], int)
            assert results[thread_id] == reads_per_thread  # All reads should succeed
    
    def test_mixed_concurrent_operations(self, temp_cache_dir, sample_datapoint):
        """Test mixed concurrent put/get operations."""
        cache = DiskDatasetCache(
            cache_dir=temp_cache_dir,
            version_hash="mixed_concurrent_test",
            enable_validation=False
        )
        
        results = {}
        
        def writer_worker(worker_id):
            """Writer worker function."""
            try:
                for i in range(10):
                    idx = worker_id * 10 + i
                    cache.put(idx, sample_datapoint)
                results[f'writer_{worker_id}'] = 'success'
            except Exception as e:
                results[f'writer_{worker_id}'] = f'error: {e}'
        
        def reader_worker(worker_id):
            """Reader worker function."""
            try:
                successful_reads = 0
                for i in range(20):
                    idx = i % 20  # Try to read from range that writers are filling
                    result = cache.get(idx)
                    if result is not None:
                        successful_reads += 1
                    time.sleep(0.001)  # Small delay to allow for more interesting interleavings
                results[f'reader_{worker_id}'] = successful_reads
            except Exception as e:
                results[f'reader_{worker_id}'] = f'error: {e}'
        
        # Start mixed threads
        threads = []
        
        # Start writers
        for i in range(2):
            thread = threading.Thread(target=writer_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Start readers
        for i in range(3):
            thread = threading.Thread(target=reader_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify writer results
        for i in range(2):
            assert results[f'writer_{i}'] == 'success'
        
        # Verify reader results (should not error, may have varying success counts)
        for i in range(3):
            assert isinstance(results[f'reader_{i}'], int)
            assert results[f'reader_{i}'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])