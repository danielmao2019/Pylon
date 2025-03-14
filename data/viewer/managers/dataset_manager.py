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
import logging

from data.viewer.managers.dataset_cache import DatasetCache
from data.viewer.managers.dataset_loader import DatasetLoader
from data.viewer.managers.transform_manager import TransformManager

# Mapping of dataset names to whether they are 3D datasets
THREE_D_DATASETS = {
    'urb3dcd': True,  # Urb3DCDDataset
    'slpccd': True,   # SLPCCDDataset
}


class DatasetManager:
    """Manages dataset operations including loading, caching, and transformations."""

    def __init__(self, config_dir: Optional[str] = None, cache_size: int = 100, cache_memory_mb: float = 1000):
        """Initialize the dataset manager.
        
        Args:
            config_dir: Optional directory containing dataset configurations
            cache_size: Maximum number of items to cache
            cache_memory_mb: Maximum memory usage for cache in MB
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.loader = DatasetLoader(config_dir)
        self.transform_manager = TransformManager()
        self._datasets: Dict[str, Any] = {}
        self._caches: Dict[str, DatasetCache] = {}
        
        # Cache settings
        self.cache_size = cache_size
        self.cache_memory_mb = cache_memory_mb
        
        # Store configurations for easy access
        self._configs = self.loader.configs
        
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets.
        
        Returns:
            List of dataset names
        """
        return list(self._configs.keys())
        
    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset and return its information.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Dataset info dictionary
        """
        # Create cache for dataset if needed
        if dataset_name not in self._caches:
            self._caches[dataset_name] = DatasetCache(
                max_size=self.cache_size,
                max_memory_mb=self.cache_memory_mb
            )
            
        # Load dataset if not already loaded
        if dataset_name not in self._datasets:
            dataset = self.loader.load_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Failed to load dataset: {dataset_name}")
            self._datasets[dataset_name] = dataset
            
        dataset = self._datasets[dataset_name]
        
        # Get transforms from dataset config
        config = self.loader.get_config(dataset_name)
        dataset_cfg = config.get('train_dataset', {})
        transforms_cfg = dataset_cfg.get('args', {}).get('transforms_cfg', {})
        
        # Register transforms
        self.transform_manager.register_transforms_from_config(transforms_cfg)
        
        # Get dataset info
        info = {
            'name': dataset_name,
            'length': len(dataset),
            'class_labels': getattr(dataset, 'class_labels', {}),
            'available_transforms': self.transform_manager.get_transform_names(),
            'cache_stats': self._caches[dataset_name].get_stats(),
            'is_3d': THREE_D_DATASETS.get(dataset_name, False)
        }
        
        return info
        
    def get_item(self, dataset_name: str, index: int, transform_name: Optional[str] = None) -> Optional[Any]:
        """Get an item from a dataset, optionally applying a transform.
        
        Args:
            dataset_name: Name of the dataset
            index: Index of the item
            transform_name: Optional name of transform to apply
            
        Returns:
            Dataset item or None if not found/error
        """
        if dataset_name not in self._datasets:
            self.logger.error(f"Dataset not loaded: {dataset_name}")
            return None
            
        # Try to get from cache first
        cache = self._caches[dataset_name]
        item = cache.get(index)
        
        if item is None:
            try:
                item = self._datasets[dataset_name][index]
                cache.put(index, item)
            except Exception as e:
                self.logger.error(f"Error getting item {index} from dataset {dataset_name}: {str(e)}")
                return None
                
        # Apply transform if requested
        if transform_name is not None:
            item = self.transform_manager.apply_transform(transform_name, item)
            
        return item
        
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear cache for a dataset or all datasets.
        
        Args:
            dataset_name: Optional name of dataset to clear cache for
        """
        if dataset_name is not None:
            if dataset_name in self._caches:
                self._caches[dataset_name].clear()
        else:
            for cache in self._caches.values():
                cache.clear()
                
    def get_cache_stats(self, dataset_name: str) -> Optional[Dict[str, int]]:
        """Get cache statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Cache statistics or None if dataset not found
        """
        if dataset_name not in self._caches:
            return None
        return self._caches[dataset_name].get_stats()
