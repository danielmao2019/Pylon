from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import threading
import logging


class BaseCache(ABC):
    """Abstract base class for all cache implementations."""

    def __init__(self):
        """Initialize common components for all cache implementations."""
        # Initialize thread lock for thread-safe operations
        self._lock = threading.Lock()
        # Initialize logger for this cache instance
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get(self, cache_filepath: str, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve item from cache.

        Args:
            cache_filepath: Absolute filepath used as cache key
            device: Device to load tensors to (e.g., 'cuda:0', 'cpu')

        Returns:
            Cached datapoint or None if not found
        """
        raise NotImplementedError("Subclasses must implement get()")

    @abstractmethod
    def put(self, value: Dict[str, Any], cache_filepath: str) -> None:
        """Store item in cache.

        Args:
            value: Raw datapoint with 'inputs', 'labels', 'meta_info' keys
            cache_filepath: Absolute filepath used as cache key
        """
        raise NotImplementedError("Subclasses must implement put()")

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached items."""
        raise NotImplementedError("Subclasses must implement clear()")

    @abstractmethod
    def get_size(self) -> int:
        """Get number of cached items."""
        raise NotImplementedError("Subclasses must implement get_size()")

    def __getstate__(self):
        """Custom pickle serialization - exclude non-picklable thread lock.

        This method is called during pickling/multiprocessing to prepare the
        object for serialization. Thread locks cannot be pickled,
        so we remove them from the state dictionary.

        Returns:
            Dictionary containing the instance state without thread locks.
        """
        state = self.__dict__.copy()
        # Remove non-picklable thread lock if it exists
        if '_lock' in state:
            state['_lock'] = None
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization - restore state and recreate thread lock.

        This method is called during unpickling/multiprocessing to restore the
        object from its serialized state. We restore the state and create new
        thread lock for the new process.

        Args:
            state: Dictionary containing the serialized instance state.
        """
        self.__dict__.update(state)
        # Create new thread lock for this process if the subclass uses one
        if '_lock' in self.__dict__:
            self._lock = threading.Lock()
