from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseCache(ABC):
    """Abstract base class for all cache implementations."""
    
    @abstractmethod
    def get(self, idx: int, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve item from cache.
        
        Args:
            idx: Index of the datapoint
            device: Device to load tensors to (e.g., 'cuda:0', 'cpu')
            
        Returns:
            Cached datapoint or None if not found
        """
        raise NotImplementedError("Subclasses must implement get()")
    
    @abstractmethod
    def put(self, idx: int, value: Dict[str, Any]) -> None:
        """Store item in cache.
        
        Args:
            idx: Index of the datapoint
            value: Raw datapoint with 'inputs', 'labels', 'meta_info' keys
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
