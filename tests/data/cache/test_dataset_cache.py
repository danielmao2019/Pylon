import pytest
import torch
import threading
import time
import psutil
from data.cache import DatasetCache


@pytest.fixture
def sample_tensor():
    return torch.randn(3, 64, 64)


@pytest.fixture
def sample_datapoint(sample_tensor):
    return {
        'inputs': {'image': sample_tensor},
        'labels': {'class': torch.tensor([1])},
        'meta_info': {'filename': 'test.jpg'}
    }


def test_cache_initialization():
    """Test basic cache initialization with different parameters."""
    # Test default initialization
    cache = DatasetCache()
    assert cache.max_memory_percent == 80.0
    assert cache.enable_validation is True
    
    # Test custom initialization
    cache = DatasetCache(max_memory_percent=50.0, enable_validation=False)
    assert cache.max_memory_percent == 50.0
    assert cache.enable_validation is False
    
    # Test initial state
    assert len(cache.cache) == 0
    assert len(cache.checksums) == 0
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.validation_failures == 0


def test_invalid_initialization():
    """Test initialization with invalid parameters."""
    # Test negative memory percentage
    with pytest.raises(ValueError):
        DatasetCache(max_memory_percent=-1.0)
    
    # Test memory percentage > 100
    with pytest.raises(ValueError):
        DatasetCache(max_memory_percent=101.0)


def test_cache_put_and_get(sample_datapoint):
    """Test basic put and get operations."""
    cache = DatasetCache()
    
    # Test put
    cache.put(0, sample_datapoint)
    assert len(cache.cache) == 1
    
    # Test get
    retrieved = cache.get(0)
    assert retrieved is not None
    assert retrieved['inputs']['image'].shape == sample_datapoint['inputs']['image'].shape
    assert torch.all(retrieved['inputs']['image'] == sample_datapoint['inputs']['image'])
    assert retrieved['labels']['class'] == sample_datapoint['labels']['class']
    assert retrieved['meta_info']['filename'] == sample_datapoint['meta_info']['filename']
    
    # Test get with non-existent key
    assert cache.get(1) is None


def test_cache_memory_management():
    """Test memory management and eviction."""
    initial_memory = psutil.Process().memory_percent()
    cache = DatasetCache(max_memory_percent=initial_memory + 0.1)  # Set threshold just above current usage
    
    # Create a reference to track the first item
    first_tensor = torch.randn(1000, 1000)  # ~4MB
    first_data = {'data': first_tensor.clone()}
    cache.put(0, first_data)
    
    # Add more items until we exceed the memory threshold
    item_count = 1
    max_items = 100  # Safety limit
    while psutil.Process().memory_percent() < cache.max_memory_percent and item_count < max_items:
        cache.put(item_count, {'data': torch.randn(1000, 1000)})
        item_count += 1
    
    # Force garbage collection to ensure memory stats are accurate
    import gc
    gc.collect()
    
    # Verify that some items were evicted (at least the first item should be gone)
    assert 0 not in cache.cache, "First item should have been evicted"
    assert len(cache.cache) < item_count, "Some items should have been evicted"
    
    # Verify memory usage is under the threshold
    assert psutil.Process().memory_percent() <= cache.max_memory_percent


def test_cache_lru_behavior(sample_datapoint):
    """Test Least Recently Used (LRU) behavior."""
    cache = DatasetCache()
    
    # Add multiple items
    for i in range(3):
        cache.put(i, sample_datapoint)
    
    # Access items in specific order
    access_order = [0, 2, 1, 2, 0]
    for i in access_order:
        cache.get(i)
    
    # Force eviction by setting low memory limit
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    
    # Add new item to trigger eviction
    cache.put(3, sample_datapoint)
    
    # Item 1 should be evicted as it was least recently used
    assert 1 not in cache.cache
    assert 0 in cache.cache
    assert 2 in cache.cache


def test_cache_thread_safety(sample_datapoint):
    """Test thread safety of cache operations."""
    cache = DatasetCache()
    num_threads = 10
    ops_per_thread = 100
    errors = []
    
    def worker(thread_id):
        try:
            for i in range(ops_per_thread):
                key = (thread_id * ops_per_thread + i) % 5
                if i % 2 == 0:
                    cache.put(key, sample_datapoint)
                else:
                    result = cache.get(key)
                    if result is not None:
                        # Verify data integrity
                        assert torch.all(result['inputs']['image'] == sample_datapoint['inputs']['image'])
        except Exception as e:
            errors.append(f"Thread {thread_id} error: {str(e)}")
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    assert not errors, f"Thread errors occurred: {errors}"
    
    # Verify cache is in valid state
    assert len(cache.cache) <= 5
    stats = cache.get_stats()
    assert stats['hits'] + stats['misses'] == (num_threads * ops_per_thread) // 2


def test_cache_validation(sample_datapoint):
    """Test cache validation mechanism."""
    cache = DatasetCache(enable_validation=True)
    
    # Test normal validation
    cache.put(0, sample_datapoint)
    assert cache.get(0) is not None
    
    # Test validation failure by corrupting cached data
    cache.cache[0]['inputs']['image'] += 1.0  # Modify tensor in-place
    assert cache.get(0) is None  # Should fail validation and return None
    assert cache.validation_failures == 1


def test_cache_stats(sample_datapoint):
    """Test cache statistics."""
    cache = DatasetCache()
    
    # Initial stats
    stats = cache.get_stats()
    assert stats['size'] == 0
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['hit_rate'] == 0
    
    # Add and access items
    cache.put(0, sample_datapoint)
    cache.get(0)  # Hit
    cache.get(1)  # Miss
    
    stats = cache.get_stats()
    assert stats['size'] == 1
    assert stats['hits'] == 1
    assert stats['misses'] == 1
    assert abs(stats['hit_rate'] - 0.5) < 1.0e-05


def test_cache_deep_copy_isolation(sample_datapoint):
    """Test that cached items are properly isolated through deep copying."""
    cache = DatasetCache()
    
    # Put item in cache
    cache.put(0, sample_datapoint)
    
    # Modify original data
    sample_datapoint['inputs']['image'] += 1.0
    sample_datapoint['meta_info']['filename'] = 'modified.jpg'
    
    # Get cached item and verify it's unchanged
    cached = cache.get(0)
    assert not torch.all(cached['inputs']['image'] == sample_datapoint['inputs']['image'])
    assert cached['meta_info']['filename'] == 'test.jpg'
    
    # Modify retrieved data and verify cache is unchanged
    retrieved = cache.get(0)
    retrieved['inputs']['image'] += 1.0
    retrieved['meta_info']['filename'] = 'modified2.jpg'
    
    cached_again = cache.get(0)
    assert not torch.all(cached_again['inputs']['image'] == retrieved['inputs']['image'])
    assert cached_again['meta_info']['filename'] == 'test.jpg'


def test_max_cache_size():
    """Test that cache respects maximum size limits."""
    cache = DatasetCache(max_memory_percent=99.9)  # High threshold
    max_size = 1000
    cache.max_size = max_size  # Set maximum number of items
    
    # Add items up to and beyond max size
    for i in range(max_size + 100):
        cache.put(i, {'data': torch.randn(10, 10)})
        assert len(cache.cache) <= max_size, f"Cache exceeded max size: {len(cache.cache)} > {max_size}"
