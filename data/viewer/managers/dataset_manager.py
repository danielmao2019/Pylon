"""Dataset management for the viewer.

This module contains the DatasetManager class which handles loading, caching,
and managing datasets for the viewer.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
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
    'synth_pcr_dataset': True,  # Synthetic PCR dataset
    'real_pcr_dataset': True,   # Real PCR dataset
}

# Mapping of dataset types to their format specifications
DATASET_FORMATS = {
    'change_detection': {
        'input_format': {
            'image': ['img_1', 'img_2'],
            'point_cloud': ['pc_1', 'pc_2']
        },
        'label_format': ['change_map']
    },
    'point_cloud_registration': {
        'input_format': {
            'point_cloud': ['src_pc', 'tgt_pc'],
            'optional': ['correspondences']
        },
        'label_format': ['transform']
    }
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
        
        # Determine dataset type from name
        dataset_type = dataset_name.split('/')[0] if '/' in dataset_name else 'change_detection'
        dataset_format = DATASET_FORMATS.get(dataset_type, DATASET_FORMATS['change_detection'])
        
        # Get base dataset name without type prefix
        base_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
        
        # Get dataset info
        info = {
            'name': dataset_name,
            'type': dataset_type,
            'length': len(dataset),
            'class_labels': getattr(dataset, 'class_labels', {}),
            'transforms': self.transform_manager.get_available_transforms(),
            'cache_stats': self._caches[dataset_name].get_stats(),
            'is_3d': THREE_D_DATASETS.get(base_name, False),
            'input_format': dataset_format['input_format'],
            'label_format': dataset_format['label_format']
        }
        
        return info

    def _load_raw_datapoint(self, dataset_name: str, index: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load raw datapoint without any transforms.
        
        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            
        Returns:
            Dict with structure:
            {
                'inputs': Dict[str, torch.Tensor],
                'labels': Dict[str, torch.Tensor],
                'meta_info': Dict[str, Any]
            }
            or None if loading fails
        """
        dataset = self._datasets.get(dataset_name)
        if dataset is None:
            return None
            
        try:
            inputs, labels, meta_info = dataset._load_datapoint(index)
            return {
                'inputs': inputs,
                'labels': labels,
                'meta_info': meta_info
            }
        except Exception as e:
            self.logger.error(f"Error loading datapoint {index} from dataset {dataset_name}: {str(e)}")
            return None
        
    def get_datapoint(self, dataset_name: str, index: int, transform_indices: Optional[List[int]] = None) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get datapoint with optional transforms applied.
        
        Args:
            dataset_name: Name of dataset
            index: Index of datapoint
            transform_indices: Optional list of transform indices to apply
            
        Returns:
            Dict containing inputs, labels and meta_info, with transforms applied if specified
        """
        if dataset_name not in self._datasets:
            self.logger.error(f"Dataset not loaded: {dataset_name}")
            return None
            
        # Try to get from cache first
        cache_key = (index, tuple(transform_indices) if transform_indices else None)
        cache = self._caches.get(dataset_name)
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Load raw datapoint
        datapoint = self._load_raw_datapoint(dataset_name, index)
            
        # Apply transforms if specified
        if transform_indices:
            datapoint = self.transform_manager.apply_transforms(datapoint, transform_indices)
            
        # Cache and return
        if cache is not None:
            cache.put(cache_key, datapoint)
        return datapoint
        
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
