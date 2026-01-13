import os
from datetime import datetime
from typing import Any, Dict, Optional, Set

import torch
import xxhash

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
        dataset_class_name: Optional[str] = None,
        version_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            cache_dir: Base cache directory (e.g., "/path/to/data_cache")
            version_hash: Hash identifying this dataset version
            enable_validation: Whether to enable checksum validation (default: False for performance)
            dataset_class_name: Name of the dataset class for metadata
            version_dict: Dictionary containing version parameters for metadata
        """
        super().__init__()

        self.cache_dir = cache_dir
        self.version_hash = version_hash
        self.enable_validation = enable_validation
        self.dataset_class_name = dataset_class_name
        self.version_dict = version_dict

        self.validated_keys: Set[str] = set()

        os.makedirs(cache_dir, exist_ok=True)
        self.version_dir = os.path.join(cache_dir, version_hash)
        os.makedirs(self.version_dir, exist_ok=True)

        self.metadata_file = os.path.join(cache_dir, 'cache_metadata.json')
        self._update_metadata()

    def _compute_checksum(self, value: Dict[str, Any]) -> str:
        """Compute a fast checksum for a cached item using xxhash."""

        def prepare_for_hash(item):
            if isinstance(item, torch.Tensor):
                return item.cpu().numpy().tobytes()
            if isinstance(item, dict):
                return {k: prepare_for_hash(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [prepare_for_hash(x) for x in item]
            return item

        hashable_data = prepare_for_hash(value)
        return xxhash.xxh64(str(hashable_data).encode()).hexdigest()

    def _validate_item(
        self, cache_key: str, value: Dict[str, Any], stored_checksum: str
    ) -> None:
        """Validate a cached item against its stored checksum (first-access-only per session)."""
        if not self.enable_validation:
            return

        if cache_key in self.validated_keys:
            return

        current_checksum = self._compute_checksum(value)
        if current_checksum != stored_checksum:
            raise ValueError("Disk cache validation failed - data corruption detected")

        self.validated_keys.add(cache_key)

    def exists(self, cache_filepath: str) -> bool:
        """Check if cache file exists for given filepath."""
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        assert os.path.isabs(
            cache_filepath
        ), f"cache_filepath must be absolute, got {cache_filepath=}"
        return os.path.exists(cache_filepath)

    def get(
        self, cache_filepath: str, device: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Thread-safe cache retrieval from disk with validation."""
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        assert os.path.isabs(
            cache_filepath
        ), f"cache_filepath must be absolute, got {cache_filepath=}"

        if not os.path.isfile(cache_filepath):
            return None

        with self._lock:
            try:
                map_location = device if device is not None else 'cpu'
                cached_data = load_torch(cache_filepath, map_location=map_location)
                value = {k: v for k, v in cached_data.items() if k != 'checksum'}

                if self.enable_validation:
                    stored_checksum = cached_data.get('checksum', '')
                    self._validate_item(cache_filepath, value, stored_checksum)

                return value

            except Exception as e:
                self.logger.warning(
                    f"Failed to load corrupted cache file {cache_filepath}: {e}"
                )
                try:
                    os.remove(cache_filepath)
                    self.logger.info(f"Removed corrupted cache file: {cache_filepath}")
                except Exception as remove_error:
                    self.logger.warning(
                        f"Failed to remove corrupted cache file: {remove_error}"
                    )
                return None

    def put(self, value: Dict[str, Any], cache_filepath: str) -> None:
        """Thread-safe cache storage to disk with checksum computation."""
        assert isinstance(
            cache_filepath, str
        ), f"cache_filepath must be str, got {type(cache_filepath)=}"
        assert os.path.isabs(
            cache_filepath
        ), f"cache_filepath must be absolute, got {cache_filepath=}"

        with self._lock:
            cached_data = {
                'inputs': value['inputs'],
                'labels': value['labels'],
                'meta_info': value['meta_info'],
            }

            if self.enable_validation:
                cached_data['checksum'] = self._compute_checksum(value)

            save_torch(cached_data, cache_filepath)

    def clear(self) -> None:
        """Clear all cache files for this version."""
        with self._lock:
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
        with self._lock:
            metadata = {}
            if os.path.exists(self.metadata_file):
                metadata = load_json(self.metadata_file)

            if self.version_hash in metadata:
                created_at = metadata[self.version_hash]['created_at']
            else:
                created_at = datetime.now()

            version_metadata = {
                'created_at': created_at,
                'cache_dir': self.cache_dir,
                'version_dir': self.version_dir,
                'enable_validation': self.enable_validation,
            }

            if self.dataset_class_name is not None:
                version_metadata['dataset_class_name'] = self.dataset_class_name

            if self.version_dict is not None:
                version_metadata['version_dict'] = self.version_dict

            metadata[self.version_hash] = version_metadata

            save_json(metadata, self.metadata_file)
