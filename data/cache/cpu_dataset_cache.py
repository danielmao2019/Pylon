from typing import Dict, Any, Optional, Set
import copy
from collections import OrderedDict

import psutil
import torch
import xxhash

from data.cache.base_cache import BaseCache
from utils.ops import apply_tensor_op


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
        # Initialize base class (sets up _lock and logger)
        super().__init__()

        if not 0 <= max_memory_percent <= 100:
            raise ValueError(
                f"max_memory_percent must be between 0 and 100, got {max_memory_percent}"
            )

        self.max_memory_percent = max_memory_percent
        self.enable_validation = enable_validation

        # Track which keys have been validated this session (first-access-only)
        self.validated_keys: Set[str] = set()

        # Initialize cache structures
        self.cache = OrderedDict()  # LRU cache with key -> value mapping
        self.checksums = {}  # key -> checksum mapping for validation
        self.memory_usage = {}  # key -> memory usage in bytes
        self.total_memory = 0  # Total memory usage in bytes

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

    def _validate_item(self, cache_key: str, value: Dict[str, Any]) -> bool:
        """Validate a cached item against its stored checksum (first-access-only per session)."""
        if not self.enable_validation:
            return True

        # Only validate on first access per session
        if cache_key in self.validated_keys:
            return True

        if cache_key not in self.checksums:
            return False

        current_checksum = self._compute_checksum(value)
        is_valid = current_checksum == self.checksums[cache_key]

        if not is_valid:
            raise ValueError(
                f"Cache validation failed for key {cache_key} - data corruption detected"
            )

        # Mark as validated for this session
        self.validated_keys.add(cache_key)
        return is_valid

    def get(
        self, cache_filepath: str, device: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval with LRU update and validation."""
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        with self._lock:
            if cache_filepath in self.cache:
                value = self.cache[cache_filepath]

                if self._validate_item(cache_filepath, value):
                    # Update LRU order
                    self.cache.move_to_end(cache_filepath)
                    self.logger.debug(
                        f"Current LRU order after get({cache_filepath}): {list(self.cache.keys())}"
                    )
                    # CPU cache always returns data on CPU, no device transfer
                    return copy.deepcopy(value)
                else:
                    # Remove invalid item
                    self.logger.debug(
                        f"Removing invalid key {cache_filepath} from cache"
                    )
                    self.cache.pop(cache_filepath)
                    self.checksums.pop(cache_filepath, None)

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

    def put(self, value: Dict[str, Any], cache_filepath: str) -> None:
        """Thread-safe cache insertion with memory management and checksum computation."""
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        with self._lock:
            # Ensure all tensors are on CPU for CPU cache storage without duplicating GPU tensors
            copied_value = apply_tensor_op(method='detach', inputs=value)
            copied_value = apply_tensor_op(method='cpu', inputs=copied_value)

            new_item_memory = self._calculate_item_memory(copied_value)

            # Remove old entry if it exists
            if cache_filepath in self.cache:
                old_memory = self.memory_usage.pop(cache_filepath)
                self.total_memory -= old_memory
                self.cache.pop(cache_filepath)
                self.checksums.pop(cache_filepath, None)

            # Calculate current max cache size in bytes
            max_cache_bytes = int(
                (self.max_memory_percent / 100.0) * psutil.virtual_memory().total
            )

            # Evict items if needed to stay under memory limit
            while self.cache and (
                self.total_memory + new_item_memory > max_cache_bytes
            ):
                # Get the least recently used key
                evict_key = next(iter(self.cache))
                evicted_memory = self.memory_usage.pop(evict_key)
                self.total_memory -= evicted_memory
                self.cache.pop(evict_key)
                self.checksums.pop(evict_key, None)

            # Store new item
            self.cache[cache_filepath] = copied_value
            self.memory_usage[cache_filepath] = new_item_memory
            self.total_memory += new_item_memory

            if self.enable_validation:
                self.checksums[cache_filepath] = self._compute_checksum(copied_value)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
            self.checksums.clear()
            self.memory_usage.clear()
            self.validated_keys.clear()
            self.total_memory = 0

    def get_size(self) -> int:
        """Get number of cached items."""
        return len(self.cache)
