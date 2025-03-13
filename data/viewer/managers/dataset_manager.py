"""Dataset management for the viewer.

This module contains the DatasetManager class which handles loading, caching,
and managing datasets for the viewer.
"""
from typing import Dict, Any, Optional, List, Tuple, Callable
import traceback
import weakref
import threading
from collections import OrderedDict
import numpy as np
from data.viewer.layout.controls.dataset import get_available_datasets

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
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Transform functions
        self._transform_functions: Dict[str, callable] = {}
        
        # Cache for each dataset
        self._caches: Dict[str, DatasetCache] = {}
        
        # Preloading settings
        self._preload_window = 5  # Number of items to preload in each direction
        
        # Load available datasets
        self._load_available_datasets()
        
    def _load_available_datasets(self) -> None:
        """Load available dataset configurations."""
        self._configs = get_available_datasets()
        
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names.
        
        Returns:
            List of dataset names
        """
        return list(self._configs.keys())
        
    def get_dataset_config(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset configuration or None if not found
        """
        return self._configs.get(dataset_name)
        
    def load_dataset(self, dataset_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Load a dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (success, message, dataset_info)
        """
        try:
            if dataset_name not in self._configs:
                return False, f"Dataset {dataset_name} not found", {}
                
            # Create cache for dataset if needed
            if dataset_name not in self._caches:
                self._caches[dataset_name] = DatasetCache()
                
            # Load dataset if not already loaded
            if dataset_name not in self._datasets:
                config = self._configs[dataset_name]
                dataset = self._load_dataset_from_config(config)
                self._datasets[dataset_name] = dataset
                
            dataset = self._datasets[dataset_name]
            
            # Get dataset info
            info = {
                'name': dataset_name,
                'length': len(dataset),
                'class_labels': getattr(dataset, 'class_labels', {}),
                'is_3d': THREE_D_DATASETS.get(dataset_name, False),
                'available_transforms': self._get_available_transforms(dataset)
            }
            
            return True, "Dataset loaded successfully", info
            
        except Exception as e:
            return False, str(e), {}
            
    def _load_dataset_from_config(self, config: Dict[str, Any]) -> Optional[Any]:
        """Load a dataset from its configuration.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Loaded dataset or None if loading failed
        """
        try:
            # Import dataset class
            dataset_class = config.get('class')
            if not dataset_class:
                return None
            
            # Get dataset arguments
            args = config.get('args', {})
            
            # Create dataset instance
            dataset = dataset_class(**args)
            
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset from config: {e}")
            return None
    
    def get_datapoint(self, dataset_name: str, index: int) -> Optional[Any]:
        """Get a datapoint from the dataset.
        
        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            
        Returns:
            Datapoint or None if not found
        """
        try:
            # Check cache first
            if dataset_name in self._caches:
                cached_item = self._caches[dataset_name].get(index)
                if cached_item is not None:
                    return cached_item
                    
            # Get dataset
            dataset = self._datasets.get(dataset_name)
            if dataset is None:
                return None
                
            # Get item
            item = dataset[index]
            
            # Cache item
            if dataset_name in self._caches:
                self._caches[dataset_name].put(index, item)
                
            # Preload adjacent items
            self._preload_adjacent_items(dataset_name, index)
            
            return item
            
        except Exception as e:
            print(f"Error getting datapoint: {e}")
            return None
            
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
                    try:
                        item = dataset[index]
                        cache.put(index, item)
                    except Exception:
                        continue
                        
    def apply_transforms(self, dataset_name: str, index: int, transforms: List[str]) -> Optional[Any]:
        """Apply transforms to a datapoint.
        
        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            transforms: List of transform names to apply
            
        Returns:
            Transformed datapoint or None if error
        """
        try:
            # Get original datapoint
            datapoint = self.get_datapoint(dataset_name, index)
            if datapoint is None:
                return None
                
            # Apply transforms
            for transform in transforms:
                if transform in self._transform_functions:
                    datapoint = self._transform_functions[transform](datapoint)
                    
            return datapoint
            
        except Exception as e:
            print(f"Error applying transforms: {e}")
            return None
            
    def _get_available_transforms(self, dataset: Any) -> List[str]:
        """Get list of available transforms for a dataset.
        
        Args:
            dataset: Dataset object
            
        Returns:
            List of transform names
        """
        # This would be implemented based on your transform system
        return []
    
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