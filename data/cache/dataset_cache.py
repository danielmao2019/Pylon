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
        self.memory_usage = {}  # Track memory usage per item
        self.total_memory = 0  # Track total memory usage
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
            raise ValueError(f"Cache validation failed for key {key} - data corruption detected")

        return is_valid

    def get(self, key: int) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval with LRU update and validation."""
        with self.lock:
            if key in self.cache:
                value = self.cache[key]

                # Validate item before returning
                if self._validate_item(key, value):
                    self.hits += 1
                    # Update LRU order - move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.logger.debug(f"Current LRU order after get({key}): {list(self.cache.keys())}")
                    return copy.deepcopy(value)
                else:
                    # Remove invalid item
                    self.logger.debug(f"Removing invalid key {key} from cache")
                    self.cache.pop(key)
                    self.checksums.pop(key, None)

            self.misses += 1
            return None

    @staticmethod
    def _calculate_item_memory(value: Dict[str, Any]) -> int:
        """Calculate approximate memory usage of a cached item in bytes."""
        total_memory = 0

        # Calculate memory for tensors
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
        # Add overhead for Python objects, metadata, etc
        overhead = 1024  # 1KB overhead per item
        return memory + overhead

    def put(self, key: int, value: Dict[str, Any]) -> None:
        """Thread-safe cache insertion with memory management and checksum computation."""
        with self.lock:
            # Make a copy of the value and calculate its memory
            copied_value = copy.deepcopy(value)
            new_item_memory = self._calculate_item_memory(copied_value)

            # Remove old entry if it exists
            if key in self.cache:
                self.logger.debug(f"Removing old entry for key {key}")
                old_memory = self.memory_usage.pop(key)
                self.total_memory -= old_memory
                self.cache.pop(key)
                self.checksums.pop(key, None)

            # Get current system memory limit in bytes
            total_system_memory = psutil.virtual_memory().total
            max_cache_memory = (self.max_memory_percent / 100.0) * total_system_memory

            # Evict items if needed to stay under memory limit
            while self.cache and (self.total_memory + new_item_memory > max_cache_memory):
                # Get the least recently used key
                evict_key = next(iter(self.cache))
                evicted_memory = self.memory_usage.pop(evict_key)
                self.total_memory -= evicted_memory
                self.cache.pop(evict_key)
                self.checksums.pop(evict_key, None)
                self.logger.debug(f"Evicted LRU key {evict_key} to free {evicted_memory} bytes")

            # Store new item
            self.cache[key] = copied_value
            self.memory_usage[key] = new_item_memory
            self.total_memory += new_item_memory

            if self.enable_validation:
                self.checksums[key] = self._compute_checksum(copied_value)

            self.logger.debug(f"Current LRU order after put({key}): {list(self.cache.keys())}")

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
