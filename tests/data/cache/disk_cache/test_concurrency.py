"""Test disk cache concurrent access patterns and thread safety."""

import pytest
import tempfile
import time
import threading
import torch
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


def test_concurrent_put_operations(temp_cache_dir, sample_datapoint):
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


def test_concurrent_get_operations(temp_cache_dir, sample_datapoint):
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


def test_mixed_concurrent_operations(temp_cache_dir, sample_datapoint):
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


def test_concurrent_validation_operations(temp_cache_dir, sample_datapoint):
    """Test concurrent operations with validation enabled."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="concurrent_validation_test",
        enable_validation=True
    )
    
    # Pre-populate with validation
    for i in range(10):
        cache.put(i, sample_datapoint)
    
    results = {}
    
    def validation_worker(worker_id):
        """Worker that reads with validation."""
        try:
            successful_reads = 0
            for i in range(20):
                idx = i % 10
                result = cache.get(idx)
                if result is not None:
                    successful_reads += 1
                time.sleep(0.001)
            results[f'validator_{worker_id}'] = successful_reads
        except Exception as e:
            results[f'validator_{worker_id}'] = f'error: {e}'
    
    # Start multiple validation workers
    threads = []
    for i in range(5):
        thread = threading.Thread(target=validation_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # All operations should succeed
    for i in range(5):
        assert isinstance(results[f'validator_{i}'], int)
        assert results[f'validator_{i}'] == 20


def test_concurrent_clear_operations(temp_cache_dir, sample_datapoint):
    """Test concurrent clear operations with other operations."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="concurrent_clear_test",
        enable_validation=False
    )
    
    results = {}
    
    def writer_worker():
        """Continuously write to cache."""
        try:
            for i in range(100):
                cache.put(i % 20, sample_datapoint)
                time.sleep(0.001)
            results['writer'] = 'success'
        except Exception as e:
            results['writer'] = f'error: {e}'
    
    def reader_worker():
        """Continuously read from cache."""
        try:
            successful_reads = 0
            for i in range(100):
                result = cache.get(i % 20)
                if result is not None:
                    successful_reads += 1
                time.sleep(0.001)
            results['reader'] = successful_reads
        except Exception as e:
            results['reader'] = f'error: {e}'
    
    def cleaner_worker():
        """Occasionally clear cache."""
        try:
            time.sleep(0.05)  # Let some data accumulate
            cache.clear()
            time.sleep(0.05)
            cache.clear()  # Clear again
            results['cleaner'] = 'success'
        except Exception as e:
            results['cleaner'] = f'error: {e}'
    
    # Start all workers
    threads = [
        threading.Thread(target=writer_worker),
        threading.Thread(target=reader_worker),
        threading.Thread(target=cleaner_worker)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify no exceptions occurred
    assert results['writer'] == 'success'
    assert isinstance(results['reader'], int)
    assert results['reader'] >= 0
    assert results['cleaner'] == 'success'


def test_thread_lock_functionality(temp_cache_dir, sample_datapoint):
    """Test that thread locks work correctly."""
    cache = DiskDatasetCache(
        cache_dir=temp_cache_dir,
        version_hash="lock_test",
        enable_validation=False
    )
    
    # Test that lock exists and is functional
    assert hasattr(cache, 'lock')
    assert cache.lock is not None
    
    # Test lock acquisition without calling methods that also use the lock
    with cache.lock:
        # Should be able to acquire lock - just test the lock mechanism
        pass
    
    # Test that operations work normally (they handle their own locking)
    cache.put(0, sample_datapoint)
    
    # Test that operations work after lock release
    result = cache.get(0)
    assert result is not None
    
    # Test concurrent lock usage
    results = {}
    
    def lock_worker(worker_id):
        """Worker that uses the lock."""
        try:
            # Test lock acquisition without calling methods that also use the lock
            with cache.lock:
                # Just verify we can acquire the lock and do some basic operations
                time.sleep(0.001)  # Hold lock briefly
            
            # Now do actual cache operations outside the manual lock
            cache.put(worker_id, sample_datapoint)
            result = cache.get(worker_id)
            assert result is not None
            results[worker_id] = 'success'
        except Exception as e:
            results[worker_id] = f'error: {e}'
    
    # Start multiple workers
    threads = []
    for i in range(5):
        thread = threading.Thread(target=lock_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # All should succeed
    for i in range(5):
        assert results[i] == 'success'