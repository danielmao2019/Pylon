"""Dataset caching implementation."""
from collections import OrderedDict
import threading
from typing import Dict, Any, Optional
import numpy as np
import logging

class DatasetCache:
    """LRU cache for dataset items with memory limits."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 1000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of items to cache
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[int, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        self.logger = logging.getLogger(__name__)

    def get(self, key: int) -> Optional[Any]:
        """Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found
        """
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats['hits'] += 1
                return value
            self.stats['misses'] += 1
            return None

    def put(self, key: int, value: Any) -> None:
        """Put an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Remove if exists
            if key in self.cache:
                self.cache.pop(key)

            # Add new item
            self.cache[key] = value

            # Update memory usage
            self.stats['memory_usage'] = self._get_memory_usage()

            # Trim if needed
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1

            # Check memory usage
            while self.stats['memory_usage'] > self.max_memory_bytes:
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1
                self.stats['memory_usage'] = self._get_memory_usage()

            if self.stats['memory_usage'] > self.max_memory_bytes * 0.8:
                self.logger.warning(f"Cache memory usage at {self.stats['memory_usage']/1024/1024:.2f}MB")

    def _get_memory_usage(self) -> int:
        """Get current memory usage of cache in bytes."""
        total_size = 0
        for value in self.cache.values():
            if isinstance(value, np.ndarray):
                total_size += value.nbytes
            else:
                # Rough estimate for other types
                total_size += len(str(value))
        return total_size

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self.cache.clear()
            self.stats['memory_usage'] = 0
            self.stats['evictions'] = 0
            self.stats['hits'] = 0
            self.stats['misses'] = 0

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return dict(self.stats) 