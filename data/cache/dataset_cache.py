from typing import Dict, Any, Optional
import copy
import threading
import logging
import hashlib
from collections import OrderedDict
import torch
import psutil


class DatasetCache:
    """A thread-safe cache manager for dataset items with LRU eviction policy."""
    
    def __init__(
        self,
        max_memory_percent: float = 80.0,
        enable_validation: bool = True,
    ):
        """
        Args:
            max_memory_percent (float): Maximum percentage of system memory to use (0-100)
            enable_validation (bool): Whether to enable checksum validation
            
        Raises:
            ValueError: If max_memory_percent is not between 0 and 100
        """
        if not 0 <= max_memory_percent <= 100:
            raise ValueError(f"max_memory_percent must be between 0 and 100, got {max_memory_percent}")
            
        self.max_memory_percent = max_memory_percent
        self.enable_validation = enable_validation
        
        # Initialize cache structures
        self.cache = OrderedDict()
        self.checksums = {}  # Store checksums for validation
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.validation_failures = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _compute_checksum(self, value: Dict[str, Any]) -> str:
        """Compute a checksum for a cached item."""
        # Convert tensors to numpy for consistent hashing
        def prepare_for_hash(item):
            if isinstance(item, torch.Tensor):
                return item.cpu().numpy().tobytes()
            elif isinstance(item, dict):
                return {k: prepare_for_hash(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [prepare_for_hash(x) for x in item]
            return item

        # Prepare data for hashing
        hashable_data = prepare_for_hash(value)
        return hashlib.sha256(str(hashable_data).encode()).hexdigest()

    def _validate_item(self, key: int, value: Dict[str, Any]) -> bool:
        """Validate a cached item against its stored checksum."""
        if not self.enable_validation:
            return True
            
        if key not in self.checksums:
            return False
            
        current_checksum = self._compute_checksum(value)
        is_valid = current_checksum == self.checksums[key]
        
        if not is_valid:
            self.validation_failures += 1
            self.logger.warning(f"Cache validation failed for key {key}")
            
        return is_valid

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval with LRU update and validation."""
        with self.lock:
            if key in self.cache:
                value = self.cache[key]
                
                # Validate item before returning
                if self._validate_item(key, value):
                    self.hits += 1
                    # Update LRU order
                    self.cache.pop(key)
                    self.cache[key] = value
                    return copy.deepcopy(value)
                else:
                    # Remove invalid item
                    self.cache.pop(key)
                    self.checksums.pop(key, None)
                    
            self.misses += 1
            return None
            
    def put(self, key: int, value: Dict[str, Any]) -> None:
        """Thread-safe cache insertion with memory management and checksum computation.
        Makes a deep copy of the value before storing.
        
        Args:
            key (int): The key to store the value under
            value (Dict[str, Any]): The value to store. Will be deep copied before storage.
        """
        with self.lock:
            # Check memory usage and evict if needed
            while self._get_memory_usage() > self.max_memory_percent and self.cache:
                evicted_key, _ = self.cache.popitem(last=False)  # Remove oldest item
                self.checksums.pop(evicted_key, None)
                
            # Store item and its checksum (with deep copy)
            self.cache[key] = copy.deepcopy(value)
            if self.enable_validation:
                self.checksums[key] = self._compute_checksum(self.cache[key])
                
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.Process().memory_percent()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including validation metrics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses + 1e-6),
                "memory_usage": self._get_memory_usage(),
                "validation_failures": self.validation_failures
            } 