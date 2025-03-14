"""Dataset caching for the viewer."""
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import sys


class DatasetCache:
    """Cache for dataset items with transform awareness."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 1000):
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of items to cache
            max_memory_mb: Maximum memory usage in MB
        """
        self._cache: OrderedDict[Tuple[int, Optional[Tuple[str, ...]]], Dict[str, Dict[str, Any]]] = OrderedDict()
        self._max_size = max_size
        self._max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self._current_memory = 0
        
    def get(self, key: Tuple[int, Optional[Tuple[str, ...]]]) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get an item from the cache.
        
        Args:
            key: Tuple of (index, transform_names)
            
        Returns:
            Cached datapoint or None if not found
        """
        if key in self._cache:
            value = self._cache[key]
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value
        return None
        
    def put(self, key: Tuple[int, Optional[Tuple[str, ...]]],
            value: Dict[str, Dict[str, Any]]) -> None:
        """Add an item to the cache.
        
        Args:
            key: Tuple of (index, transform_names)
            value: Datapoint to cache
        """
        # Estimate memory usage of new item
        memory_size = sys.getsizeof(value)
        
        # Remove oldest items if needed
        while (len(self._cache) >= self._max_size or 
               self._current_memory + memory_size > self._max_memory) and self._cache:
            _, removed_value = self._cache.popitem(last=False)
            self._current_memory -= sys.getsizeof(removed_value)
            
        # Add new item
        self._cache[key] = value
        self._current_memory += memory_size
        
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._current_memory = 0
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': len(self._cache),
            'memory_bytes': self._current_memory,
            'max_size': self._max_size,
            'max_memory_bytes': self._max_memory
        }
