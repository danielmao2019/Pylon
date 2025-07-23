from typing import Dict, Any, Optional, Set
import copy
import threading
import logging
import xxhash
from collections import OrderedDict
import torch
import psutil
from data.cache.base_cache import BaseCache


class CPUDatasetCache(BaseCache):
    """A thread-safe CPU memory cache manager for dataset items with LRU eviction policy."""

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
        
        # Track which keys have been validated this session (first-access-only)
        self.validated_keys: Set[int] = set()

        # Initialize cache structures
        self.cache = OrderedDict()  # LRU cache with key -> value mapping
        self.checksums = {}  # key -> checksum mapping for validation
        self.memory_usage = {}  # key -> memory usage in bytes
        self.total_memory = 0  # Total memory usage in bytes


        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize lock
        self._init_lock()

    def _init_lock(self):
        """Initialize the thread lock. Called both at init and after unpickling."""
        self.lock = threading.Lock()

    def __getstate__(self):
        """Get object state for pickling, excluding the lock."""
        state = self.__dict__.copy()
        # Don't pickle the lock or logger
        del state['lock']
        del state['logger']
        return state

    def __setstate__(self, state):
        """Restore object state from pickling and recreate the lock."""
        self.__dict__.update(state)
        # Restore unpicklable objects
        self._init_lock()
        self.logger = logging.getLogger(__name__)

    def __deepcopy__(self, memo):
        """Implement deep copy support."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Deep copy all attributes except lock and logger
        for k, v in self.__dict__.items():
            if k not in ('lock', 'logger'):
                setattr(result, k, copy.deepcopy(v, memo))

        # Initialize new lock and logger
        result._init_lock()
        result.logger = logging.getLogger(__name__)
        return result

    def _compute_checksum(self, value: Dict[str, Any]) -> str:
        """Compute a fast checksum for a cached item using xxhash."""
        def prepare_for_hash(item):
            if isinstance(item, torch.Tensor):
                return item.cpu().numpy().tobytes()
            elif isinstance(item, dict):
                return {k: prepare_for_hash(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple)):
                return [prepare_for_hash(x) for x in item]
            return item

        hashable_data = prepare_for_hash(value)
        return xxhash.xxh64(str(hashable_data).encode()).hexdigest()

    def _validate_item(self, idx: int, value: Dict[str, Any]) -> bool:
        """Validate a cached item against its stored checksum (first-access-only per session)."""
        if not self.enable_validation:
            return True
        
        # Only validate on first access per session
        if idx in self.validated_keys:
            return True

        if idx not in self.checksums:
            return False

        current_checksum = self._compute_checksum(value)
        is_valid = current_checksum == self.checksums[idx]

        if not is_valid:
            raise ValueError(f"Cache validation failed for idx {idx} - data corruption detected")
        
        # Mark as validated for this session
        self.validated_keys.add(idx)
        return is_valid

    def get(self, idx: int, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval with LRU update and validation."""
        with self.lock:
            if idx in self.cache:
                value = self.cache[idx]

                if self._validate_item(idx, value):
                    # Update LRU order
                    self.cache.move_to_end(idx)
                    self.logger.debug(f"Current LRU order after get({idx}): {list(self.cache.keys())}")
                    result = copy.deepcopy(value)
                    
                    # Move tensors to specified device if requested
                    if device is not None:
                        from utils.ops import apply_tensor_op
                        result = apply_tensor_op(func=lambda x: x.to(device), inputs=result)
                    
                    return result
                else:
                    # Remove invalid item
                    self.logger.debug(f"Removing invalid idx {idx} from cache")
                    self.cache.pop(idx)
                    self.checksums.pop(idx, None)

            return None

    @staticmethod
    def _calculate_item_memory(value: Dict[str, Any]) -> int:
        """Calculate approximate memory usage of a cached item in bytes."""
        def process_item(item):
            if isinstance(item, torch.Tensor):
                # Calculate tensor memory (size in bytes)
                return item.numel() * item.element_size()
            elif isinstance(item, dict):
                return sum(process_item(v) for v in item.values())
            elif isinstance(item, (list, tuple)):
                return sum(process_item(x) for x in item)
            return 0  # Other types negligible

        memory = process_item(value)
        overhead = 1024  # 1KB overhead for Python objects
        return memory + overhead

    def put(self, idx: int, value: Dict[str, Any]) -> None:
        """Thread-safe cache insertion with memory management and checksum computation."""
        with self.lock:
            # Make a copy and calculate its memory
            copied_value = copy.deepcopy(value)
            new_item_memory = self._calculate_item_memory(copied_value)

            # Remove old entry if it exists
            if idx in self.cache:
                old_memory = self.memory_usage.pop(idx)
                self.total_memory -= old_memory
                self.cache.pop(idx)
                self.checksums.pop(idx, None)

            # Calculate current max cache size in bytes
            max_cache_bytes = int((self.max_memory_percent / 100.0) * psutil.virtual_memory().total)

            # Evict items if needed to stay under memory limit
            while self.cache and (self.total_memory + new_item_memory > max_cache_bytes):
                # Get the least recently used key
                evict_key = next(iter(self.cache))
                evicted_memory = self.memory_usage.pop(evict_key)
                self.total_memory -= evicted_memory
                self.cache.pop(evict_key)
                self.checksums.pop(evict_key, None)

            # Store new item
            self.cache[idx] = copied_value
            self.memory_usage[idx] = new_item_memory
            self.total_memory += new_item_memory

            if self.enable_validation:
                self.checksums[idx] = self._compute_checksum(copied_value)

    def clear(self) -> None:
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.checksums.clear()
            self.memory_usage.clear()
            self.validated_keys.clear()
            self.total_memory = 0
    
    def get_size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return psutil.Process().memory_percent()
