import pytest
import torch
import threading
import time
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
    cache = DatasetCache(max_memory_percent=99.9)  # High threshold to control eviction
    
    # Add items until eviction occurs
    large_tensor = torch.randn(1000, 1000)  # Large tensor to force memory pressure
    for i in range(10):
        cache.put(i, {'data': large_tensor})
        
    # Check if older items were evicted
    assert len(cache.cache) < 10


def test_cache_lru_behavior(sample_datapoint):
    """Test Least Recently Used (LRU) behavior."""
    cache = DatasetCache()
    
    # Add multiple items
    for i in range(3):
        cache.put(i, sample_datapoint)
    
    # Access item 0 (makes it most recent)
    cache.get(0)
    
    # Force eviction of one item
    cache.max_memory_percent = 0  # Force eviction
    large_tensor = torch.randn(1000, 1000)
    cache.put(3, {'data': large_tensor})
    
    # Check if least recently used item was evicted (should be 1)
    assert 0 in cache.cache  # Most recently used
    assert 1 not in cache.cache  # Least recently used


def test_cache_thread_safety(sample_datapoint):
    """Test thread safety of cache operations."""
    cache = DatasetCache()
    num_threads = 10
    ops_per_thread = 100
    
    def worker():
        for i in range(ops_per_thread):
            key = i % 5  # Use a few keys repeatedly
            if i % 2 == 0:
                cache.put(key, sample_datapoint)
            else:
                cache.get(key)
    
    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
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
