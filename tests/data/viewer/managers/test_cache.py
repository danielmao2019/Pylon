"""Unit tests for dataset cache."""
import unittest
import numpy as np
from data.viewer.managers.dataset_cache import DatasetCache

class TestDatasetCache(unittest.TestCase):
    """Test cases for DatasetCache."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = DatasetCache(max_size=2, max_memory_mb=100)

    def test_cache_put_get(self):
        """Test basic put and get operations."""
        # Test with simple value
        self.cache.put(1, "test")
        self.assertEqual(self.cache.get(1), "test")
        
        # Test with numpy array
        arr = np.array([1, 2, 3])
        self.cache.put(2, arr)
        np.testing.assert_array_equal(self.cache.get(2), arr)

    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        self.cache.put(1, "first")
        self.cache.put(2, "second")
        self.cache.put(3, "third")  # Should evict "first"
        
        self.assertIsNone(self.cache.get(1))
        self.assertEqual(self.cache.get(2), "second")
        self.assertEqual(self.cache.get(3), "third")

    def test_cache_stats(self):
        """Test cache statistics."""
        # Initial stats
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['evictions'], 0)
        
        # Test hit
        self.cache.put(1, "test")
        self.cache.get(1)
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        
        # Test miss
        self.cache.get(2)
        stats = self.cache.get_stats()
        self.assertEqual(stats['misses'], 1)

    def test_cache_memory_limit(self):
        """Test memory limit enforcement."""
        # Create a large array that should exceed memory limit
        large_array = np.zeros((1000, 1000), dtype=np.float32)  # ~4MB
        small_array = np.zeros((10, 10), dtype=np.float32)  # ~400B
        
        self.cache = DatasetCache(max_size=10, max_memory_mb=0.1)  # 100KB limit
        
        # Put small array first
        self.cache.put(1, small_array)
        self.assertIsNotNone(self.cache.get(1))
        
        # Put large array - should evict small array due to memory limit
        self.cache.put(2, large_array)
        self.assertIsNone(self.cache.get(1))
        
    def test_cache_clear(self):
        """Test cache clear operation."""
        self.cache.put(1, "test1")
        self.cache.put(2, "test2")
        
        self.cache.clear()
        
        self.assertIsNone(self.cache.get(1))
        self.assertIsNone(self.cache.get(2))
        
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 2)
        self.assertEqual(stats['evictions'], 0)
        self.assertEqual(stats['memory_usage'], 0) 