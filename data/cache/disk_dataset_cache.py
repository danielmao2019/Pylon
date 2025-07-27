from typing import Dict, Any, Optional, Set
import os
import threading
import logging
import xxhash
import torch
from datetime import datetime
from data.cache.base_cache import BaseCache
from utils.io.json import load_json, save_json
from utils.io.torch import load_torch, save_torch


class DiskDatasetCache(BaseCache):
    """A thread-safe disk-based cache manager for dataset items with per-datapoint files."""

    def __init__(
        self,
        cache_dir: str,
        version_hash: str,
        enable_validation: bool = False,
    ):
        """
        Args:
            cache_dir: Base cache directory (e.g., "/path/to/data_cache")
            version_hash: Hash identifying this dataset version
            enable_validation: Whether to enable checksum validation (default: False for performance)
        """
        self.cache_dir = cache_dir
        self.version_hash = version_hash
        self.enable_validation = enable_validation
        
        # Track which keys have been validated this session (first-access-only)
        self.validated_keys: Set[int] = set()
        
        # Create version-specific directory
        self.version_dir = os.path.join(cache_dir, version_hash)
        os.makedirs(self.version_dir, exist_ok=True)
        
        # Metadata file path
        self.metadata_file = os.path.join(cache_dir, 'cache_metadata.json')
        
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize lock
        self._init_lock()
        
        # Update metadata
        self._update_metadata()

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

    def _get_cache_filepath(self, idx: int) -> str:
        """Get the cache file path for a given index."""
        return os.path.join(self.version_dir, f"{idx}.pt")

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

    def _validate_item(self, idx: int, value: Dict[str, Any], stored_checksum: str) -> None:
        """Validate a cached item against its stored checksum (first-access-only per session)."""
        if not self.enable_validation:
            return
        
        # Only validate on first access per session
        if idx in self.validated_keys:
            return

        current_checksum = self._compute_checksum(value)
        
        if current_checksum != stored_checksum:
            raise ValueError(f"Disk cache validation failed - data corruption detected")
        
        # Mark as validated for this session
        self.validated_keys.add(idx)

    def exists(self, idx: int) -> bool:
        """Check if cache file exists for given index."""
        return os.path.exists(self._get_cache_filepath(idx))

    def get(self, idx: int, device: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval from disk with validation."""
        cache_filepath = self._get_cache_filepath(idx)
        
        if not os.path.exists(cache_filepath):
            return None
        
        with self.lock:
            # Load from disk directly to target device if specified
            map_location = device if device is not None else 'cpu'
            cached_data = load_torch(cache_filepath, map_location=map_location)
            
            # Extract value and checksum
            value = {
                'inputs': cached_data['inputs'],
                'labels': cached_data['labels'],
                'meta_info': cached_data['meta_info'],
            }
            
            if self.enable_validation:
                stored_checksum = cached_data.get('checksum', '')
                self._validate_item(idx, value, stored_checksum)
            
            return value

    def put(self, idx: int, value: Dict[str, Any]) -> None:
        """Thread-safe cache storage to disk with checksum computation."""
        cache_filepath = self._get_cache_filepath(idx)
        
        with self.lock:
            # Prepare data for storage
            cached_data = {
                'inputs': value['inputs'],
                'labels': value['labels'],
                'meta_info': value['meta_info'],
            }
            
            if self.enable_validation:
                cached_data['checksum'] = self._compute_checksum(value)
            
            # Use atomic save function
            save_torch(cached_data, cache_filepath)

    def clear(self) -> None:
        """Clear all cache files for this version."""
        with self.lock:
            if os.path.exists(self.version_dir):
                for filename in os.listdir(self.version_dir):
                    if filename.endswith('.pt'):
                        os.remove(os.path.join(self.version_dir, filename))

    def get_size(self) -> int:
        """Get number of cached items."""
        if not os.path.exists(self.version_dir):
            return 0
        return len([f for f in os.listdir(self.version_dir) if f.endswith('.pt')])
    
    def _update_metadata(self) -> None:
        """Update cache metadata JSON file with current version info."""
        with self.lock:
            # Load existing metadata
            metadata = {}
            if os.path.exists(self.metadata_file):
                metadata = load_json(self.metadata_file)
            
            # Preserve existing created_at timestamp if this version already exists
            if self.version_hash in metadata:
                created_at = metadata[self.version_hash]['created_at']
            else:
                created_at = datetime.now()
            
            # Update with current version info
            metadata[self.version_hash] = {
                'created_at': created_at,  # datetime will be serialized by save_json
                'cache_dir': self.cache_dir,
                'version_dir': self.version_dir,
                'enable_validation': self.enable_validation,
            }
            
            # Write updated metadata
            save_json(metadata, self.metadata_file)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for all cache versions."""
        if os.path.exists(self.metadata_file):
            return load_json(self.metadata_file)
        return {}
