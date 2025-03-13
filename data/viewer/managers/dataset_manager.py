"""Dataset management for the viewer.

This module contains the DatasetManager class which handles loading, caching,
and managing datasets for the viewer.
"""
from typing import Dict, Any, Optional, List, Tuple
import traceback
from data.viewer.utils.dataset_utils import get_available_datasets, is_3d_dataset


class DatasetManager:
    """Manages dataset loading, caching, and operations.
    
    This class handles:
    - Loading and caching datasets
    - Managing dataset configurations
    - Providing dataset information
    - Handling dataset transforms
    """
    
    def __init__(self):
        """Initialize the dataset manager."""
        # Dataset cache
        self._datasets: Dict[str, Any] = {}
        
        # Dataset configurations
        self._configs: Dict[str, Dict[str, Any]] = {}
        
        # Transform functions
        self._transform_functions: Dict[str, callable] = {}
        
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
    
    def load_dataset(self, dataset_name: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Tuple of (success, message, dataset_info)
            - success: Whether loading was successful
            - message: Success/error message
            - dataset_info: Dataset information if successful, None otherwise
        """
        try:
            # Check if dataset is already loaded
            if dataset_name in self._datasets:
                dataset = self._datasets[dataset_name]
                return True, "Dataset already loaded", self._get_dataset_info(dataset_name, dataset)
            
            # Get dataset configuration
            config = self._configs.get(dataset_name)
            if config is None:
                return False, f"Dataset configuration not found: {dataset_name}", None
            
            # Load dataset
            dataset = self._load_dataset_from_config(config)
            if dataset is None:
                return False, f"Failed to load dataset: {dataset_name}", None
            
            # Cache dataset
            self._datasets[dataset_name] = dataset
            
            return True, f"Successfully loaded dataset: {dataset_name}", self._get_dataset_info(dataset_name, dataset)
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            return False, f"Error loading dataset: {str(e)}\n{error_traceback}", None
    
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
    
    def _get_dataset_info(self, dataset_name: str, dataset: Any) -> Dict[str, Any]:
        """Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset: Dataset instance
            
        Returns:
            Dictionary containing dataset information
        """
        # Get basic information
        info = {
            'name': dataset_name,
            'length': len(dataset),
            'is_3d': is_3d_dataset(dataset),
            'class_labels': {},
            'available_transforms': []
        }
        
        # Get class labels if available
        try:
            if hasattr(dataset, 'class_names'):
                info['class_labels'] = {i: name for i, name in enumerate(dataset.class_names)}
            elif hasattr(dataset, 'labels') and hasattr(dataset.labels, 'class_names'):
                info['class_labels'] = {i: name for i, name in enumerate(dataset.labels.class_names)}
        except Exception as e:
            print(f"Warning: Could not get class labels from dataset: {e}")
        
        # Get available transforms from config
        config = self._configs.get(dataset_name, {})
        train_dataset = config.get('train_dataset', {})
        if 'args' in train_dataset and 'transforms_cfg' in train_dataset['args']:
            transforms_cfg = train_dataset['args']['transforms_cfg']
            if 'args' in transforms_cfg and 'transforms' in transforms_cfg['args']:
                info['available_transforms'] = [
                    t.get('class', '') for t in transforms_cfg['args']['transforms']
                ]
        
        return info
    
    def get_dataset(self, dataset_name: str) -> Optional[Any]:
        """Get a loaded dataset by name.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset instance or None if not found
        """
        return self._datasets.get(dataset_name)
    
    def get_datapoint(self, dataset_name: str, index: int) -> Optional[Dict[str, Any]]:
        """Get a datapoint from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            
        Returns:
            Datapoint dictionary or None if not found
        """
        dataset = self.get_dataset(dataset_name)
        if dataset is None or index >= len(dataset):
            return None
            
        return dataset[index]
    
    def apply_transforms(self, dataset_name: str, index: int, 
                        transforms: List[str]) -> Optional[Dict[str, Any]]:
        """Apply transforms to a datapoint.
        
        Args:
            dataset_name: Name of the dataset
            index: Index of the datapoint
            transforms: List of transform names to apply
            
        Returns:
            Transformed datapoint or None if transformation failed
        """
        datapoint = self.get_datapoint(dataset_name, index)
        if datapoint is None:
            return None
            
        try:
            # Apply each transform
            for transform in transforms:
                if transform in self._transform_functions:
                    datapoint = self._transform_functions[transform](datapoint)
            
            return datapoint
            
        except Exception as e:
            print(f"Error applying transforms: {e}")
            return None
    
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