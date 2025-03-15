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
    tensor_dim = 1024
    num_channels = 3  # RGB images
    memory_threshold = 0.01  # 0.01% increase target
    
    # Calculate required iterations (add 3 for safety)
    tensor_bytes = tensor_dim * tensor_dim * num_channels * 4  # 4 bytes per float32
    threshold_bytes = psutil.virtual_memory().total * (memory_threshold / 100)
    required_iterations = int(threshold_bytes / tensor_bytes) + 3
    
    # Initialize cache with memory limit
    initial_memory = psutil.Process().memory_percent()
    cache = DatasetCache(max_memory_percent=initial_memory + memory_threshold)
    
    # Create tensors that will consume memory
    tensors = []
    start_memory = psutil.Process().memory_percent()
    
    for iteration in range(required_iterations):
        # Create and store tensor
        tensor = torch.randn(num_channels, tensor_dim, tensor_dim, dtype=torch.float32)
        tensors.append(tensor)  # Keep reference to prevent garbage collection
        
        datapoint = {
            'inputs': {'image': tensor},
            'labels': {'class': torch.tensor([iteration])},
            'meta_info': {'filename': f'test_{iteration}.jpg'}
        }
        cache.put(iteration, datapoint)
        
        # Check memory state
        current_memory = psutil.Process().memory_percent()
        memory_increase = current_memory - start_memory
        
        # Check if eviction occurred
        if 0 not in cache.cache:
            break
    else:
        pytest.fail(
            f"No eviction after {required_iterations + 2} iterations.\n"
            f"Memory increase: {memory_increase:.3f}%\n"
            f"Expected threshold: {memory_threshold}%\n"
            f"Cache size: {len(cache.cache)}"
        )
    
    # Verify final state
    assert len(cache.cache) > 0, "Cache should not be empty"
    assert 0 not in cache.cache, "First item should have been evicted"
    assert iteration <= required_iterations + 2, "Took too many iterations"
    
    # Cleanup
    del tensors
    import gc
    gc.collect()


def test_cache_lru_behavior(sample_datapoint):
    """Test Least Recently Used (LRU) behavior in various scenarios."""
    cache = DatasetCache()
    
    # Test 1: Basic LRU eviction order
    # Add multiple items
    for i in range(3):
        cache.put(i, sample_datapoint)
    
    # Access items in specific order to set up LRU
    access_order = [0, 2, 1, 2, 0]
    for i in access_order:
        cache.get(i)
    
    # Force eviction by setting low memory limit
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    
    # Add new item to trigger eviction
    cache.put(3, sample_datapoint)
    
    # Item 1 should be evicted as it was least recently used
    assert 1 not in cache.cache, "LRU item should have been evicted"
    assert 0 in cache.cache, "Most recently used item should be retained"
    assert 2 in cache.cache, "Second most recently used item should be retained"
    assert 3 in cache.cache, "Newly added item should be present"

    # Test 2: Re-putting existing items should update LRU order
    cache = DatasetCache()
    for i in range(3):
        cache.put(i, sample_datapoint)
    
    # Put existing item again
    cache.put(0, sample_datapoint)  # This should move 0 to most recent
    
    # Force eviction
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    cache.put(3, sample_datapoint)
    
    # Item 1 should be evicted (not 2) since 0 was moved to most recent
    assert 1 not in cache.cache, "LRU item should have been evicted"
    assert 0 in cache.cache, "Recently re-put item should be retained"
    assert 2 in cache.cache, "Non-LRU item should be retained"

    # Test 3: Multiple evictions should maintain LRU order
    cache = DatasetCache()
    for i in range(5):
        cache.put(i, sample_datapoint)
    
    # Access in specific order: [2, 0, 4, 1, 3]
    access_order = [2, 0, 4, 1, 3]
    for i in access_order:
        cache.get(i)
    
    # Force multiple evictions
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    
    # Add two new items to force two evictions
    cache.put(5, sample_datapoint)
    cache.put(6, sample_datapoint)
    
    # First two items in access order should remain (3, 1)
    assert 2 not in cache.cache, "First LRU item should be evicted"
    assert 0 not in cache.cache, "Second LRU item should be evicted"
    assert 4 in cache.cache, "More recently used item should be retained"
    assert 1 in cache.cache, "More recently used item should be retained"
    assert 3 in cache.cache, "Most recently used item should be retained"
    
    # Test 4: Get misses shouldn't affect LRU order
    cache = DatasetCache()
    for i in range(3):
        cache.put(i, sample_datapoint)
    
    # Try to get non-existent items
    cache.get(10)
    cache.get(11)
    
    # Force eviction
    cache.max_memory_percent = psutil.Process().memory_percent() - 0.1
    cache.put(3, sample_datapoint)
    
    # Original LRU order should be maintained
    assert 0 not in cache.cache, "First inserted item should be evicted"
    assert 1 in cache.cache, "Second inserted item should be retained"
    assert 2 in cache.cache, "Last inserted item should be retained"
    assert 3 in cache.cache, "Newly added item should be present"


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
