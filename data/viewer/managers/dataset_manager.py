"""Dataset management for the viewer.

This module contains the DatasetManager class which handles loading, caching,
and managing datasets for the viewer.
"""
from typing import Dict, Any, Optional, List, Tuple
import threading
from collections import OrderedDict
import numpy as np
import os
import utils.builders
import importlib.util

# Mapping of dataset names to whether they are 3D datasets
THREE_D_DATASETS = {
    'urb3dcd': True,  # Urb3DCDDataset
    'slpccd': True,   # SLPCCDDataset
}


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
                return value
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

            # Trim if needed
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            # Check memory usage
            while self._get_memory_usage() > self.max_memory_bytes:
                self.cache.popitem(last=False)

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


class DatasetManager:
    """Manages dataset loading, caching, and operations.

    This class handles:
    - Loading and caching datasets
    - Managing dataset configurations
    - Providing dataset information
    - Handling dataset transforms
    - Lazy loading of dataset items
    - Preloading of adjacent items
    """

    def __init__(self, cache_size: int = 100, cache_memory_mb: float = 1000):
        """Initialize the dataset manager.

        Args:
            cache_size: Maximum number of items to cache
            cache_memory_mb: Maximum memory usage for cache in MB
        """
        # Dataset cache
        self._datasets: Dict[str, Any] = {}

        # Dataset configurations
        self._configs = self._get_dataset_configs()
        
        # Cache for each dataset
        self._caches: Dict[str, DatasetCache] = {}

        # Preloading settings
        self._preload_window = 5  # Number of items to preload in each direction

        # Transform functions
        self._transform_functions: Dict[str, callable] = {}

    def _get_dataset_configs(self, config_dir=None):
        """Get a list of all available dataset configurations."""
        # If no config_dir is provided, use the default location
        if config_dir is None:
            # Adjust the path to be relative to the repository root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            config_dir = os.path.join(repo_root, "configs/common/datasets/change_detection/train")
        assert os.path.isdir(config_dir), f"Dataset directory not found at {config_dir}"

        dataset_configs = {}

        for file in os.listdir(config_dir):
            if file.endswith('.py') and not file.startswith('_'):
                dataset_name = file[:-3]  # Remove .py extension
                # Import the config to ensure it's valid
                spec = importlib.util.spec_from_file_location(
                    f"configs.common.datasets.change_detection.train.{dataset_name}",
                    os.path.join(config_dir, file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, 'config'):
                    # Add to the list of valid datasets
                    dataset_configs[dataset_name] = module.config

        return dataset_configs

    def get_dataset_config(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset configuration or None if not found
        """
        return self._configs.get(dataset_name)

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Dataset info dictionary
        """

        # Create cache for dataset if needed
        if dataset_name not in self._caches:
            self._caches[dataset_name] = DatasetCache()

        # Load dataset if not already loaded
        if dataset_name not in self._datasets:
            config = self._configs[dataset_name]
            dataset = self._load_dataset_from_config(config)
            self._datasets[dataset_name] = dataset

        dataset = self._datasets[dataset_name]

        # Get transforms from dataset config
        dataset_cfg = config.get('train_dataset', {})
        transforms_cfg = dataset_cfg.get('args', {}).get('transforms_cfg', {})
        transforms = transforms_cfg.get('args', {}).get('transforms', [])

        # Register transform functions
        self._transform_functions.clear()  # Clear existing transforms
        for i, (transform, _) in enumerate(transforms):
            self.register_transform(f"transform_{i}", transform)

        # Get dataset info
        info = {
            'name': dataset_name,
            'length': len(dataset),
            'class_labels': getattr(dataset, 'class_labels', {}),
            'is_3d': THREE_D_DATASETS.get(dataset_name, False),
            'available_transforms': transforms,
        }

        return info

    def _load_dataset_from_config(self, config: Dict[str, Any]) -> Any:
        """Load a dataset from its configuration.

        Args:
            config: Dataset configuration

        Returns:
            Loaded dataset
        """
        # Adjust data_root path to be relative to the repository root if needed
        dataset_cfg = config.get('train_dataset', {})
        if 'args' in dataset_cfg and 'data_root' in dataset_cfg['args'] and not os.path.isabs(dataset_cfg['args']['data_root']):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            dataset_cfg['args']['data_root'] = os.path.join(repo_root, dataset_cfg['args']['data_root'])

        # Build dataset
        dataset = utils.builders.build_from_config(dataset_cfg)
        self.logger.info(f"Loaded dataset: {dataset_cfg.get('class', 'Unknown').__name__}")
        return dataset

    def get_datapoint(self, dataset_name: str, index: int) -> Any:
        """Get a datapoint from the dataset.

        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint

        Returns:
            Datapoint
        """
        # Check cache first
        if dataset_name in self._caches:
            cached_item = self._caches[dataset_name].get(index)
            if cached_item is not None:
                return cached_item

        # Get from dataset
        dataset = self._datasets[dataset_name]
        item = dataset[index]

        # Cache the item
        if dataset_name in self._caches:
            self._caches[dataset_name].put(index, item)

        # Preload adjacent items
        self._preload_adjacent_items(dataset_name, index)

        return item

    def _preload_adjacent_items(self, dataset_name: str, current_index: int) -> None:
        """Preload items adjacent to the current index.

        Args:
            dataset_name: Name of the dataset
            current_index: Current index
        """
        if dataset_name not in self._datasets:
            return

        dataset = self._datasets[dataset_name]
        cache = self._caches.get(dataset_name)
        if cache is None:
            return

        # Preload items in a window around current_index
        for offset in range(-self._preload_window, self._preload_window + 1):
            if offset == 0:
                continue

            index = current_index + offset
            if 0 <= index < len(dataset):
                # Check if already cached
                if cache.get(index) is None:
                    item = dataset[index]
                    cache.put(index, item)

    def apply_transforms(self, dataset_name: str, index: int, transforms: List[str]) -> Any:
        """Apply transforms to a datapoint.

        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            transforms: List of transform indices to apply

        Returns:
            Transformed datapoint
        """
        # Get original datapoint
        datapoint = self.get_datapoint(dataset_name, index)

        # Apply transforms
        for transform_idx in transforms:
            transform_name = f"transform_{transform_idx}"
            if transform_name in self._transform_functions:
                transform_func = self._transform_functions[transform_name]
                datapoint = transform_func(datapoint)

        return datapoint

    def register_transform(self, name: str, transform_func: callable) -> None:
        """Register a transform function.

        Args:
            name: Name of the transform
            transform_func: Transform function to register
        """
        self._transform_functions[name] = transform_func

    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        self._datasets.clear()
