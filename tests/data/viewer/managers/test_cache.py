"""Unit tests for dataset cache."""
import pytest
import numpy as np
from data.viewer.managers.dataset_cache import DatasetCache


def get_test_cache():
    """Get a test cache instance."""
    return DatasetCache(max_size=2, max_memory_mb=100)


def get_test_data():
    """Get test data."""
    return {'data': np.zeros(1000)}  # ~8KB


def test_cache_put_get():
    """Test basic put and get operations."""
    cache = get_test_cache()
    
    # Test with simple value
    cache.put(1, "test")
    assert cache.get(1) == "test"
    
    # Test with numpy array
    arr = np.array([1, 2, 3])
    cache.put(2, arr)
    np.testing.assert_array_equal(cache.get(2), arr)


def test_cache_lru_eviction():
    """Test LRU eviction policy."""
    cache = get_test_cache()
    
    cache.put(1, "first")
    cache.put(2, "second")
    cache.put(3, "third")  # Should evict "first"
    
    assert cache.get(1) is None
    assert cache.get(2) == "second"
    assert cache.get(3) == "third"


def test_cache_stats():
    """Test cache statistics."""
    cache = get_test_cache()
    
    # Initial stats
    stats = cache.get_stats()
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['evictions'] == 0
    
    # Test hit
    cache.put(1, "test")
    cache.get(1)
    stats = cache.get_stats()
    assert stats['hits'] == 1
    
    # Test miss
    cache.get(2)
    stats = cache.get_stats()
    assert stats['misses'] == 1


def test_cache_memory_limit():
    """Test memory limit enforcement."""
    # Create a large array that should exceed memory limit
    large_array = np.zeros((1000, 1000), dtype=np.float32)  # ~4MB
    small_array = np.zeros((10, 10), dtype=np.float32)  # ~400B
    
    cache = DatasetCache(max_size=10, max_memory_mb=0.1)  # 100KB limit
    
    # Put small array first
    cache.put(1, small_array)
    assert cache.get(1) is not None
    
    # Put large array - should evict small array due to memory limit
    cache.put(2, large_array)
    assert cache.get(1) is None


def test_cache_clear():
    """Test clearing the cache."""
    cache = get_test_cache()
    test_data = get_test_data()
    
    # Add some items
    cache.put((0, None), test_data)
    cache.put((1, None), test_data)
    
    # Clear cache
    cache.clear()
    
    # Verify cache is empty
    stats = cache.get_stats()
    assert stats['size'] == 0
    assert stats['memory_bytes'] == 0
    assert stats['hits'] == 0
    assert stats['misses'] == 0
    assert stats['evictions'] == 0
