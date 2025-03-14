"""Dataset loader module."""
import os
import logging
from typing import Dict, Any, Optional
import importlib.util
from pathlib import Path

class DatasetLoader:
    """Handles loading and configuration of datasets."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the dataset loader.
        
        Args:
            config_dir: Optional directory containing dataset configurations
        """
        self.logger = logging.getLogger(__name__)
        
        # If no config_dir is provided, use the default location
        if config_dir is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            self.config_dir = os.path.join(repo_root, "configs/common/datasets/change_detection/train")
        else:
            self.config_dir = config_dir
            
        self.configs = self._load_dataset_configs()
        
    def _load_dataset_configs(self) -> Dict[str, Any]:
        """Load all available dataset configurations.
        
        Returns:
            Dictionary mapping dataset names to their configurations
        """
        dataset_configs = {}
        
        # List of supported datasets
        supported_datasets = ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd', 'urb3dcd', 'slpccd']
        
        for dataset_name in supported_datasets:
            config_file = os.path.join(self.config_dir, f"{dataset_name}.py")
            
            try:
                if not os.path.isfile(config_file):
                    self.logger.warning(f"Dataset config file not found: {config_file}")
                    continue
                    
                # Import the config
                spec = importlib.util.spec_from_file_location(
                    f"configs.common.datasets.change_detection.train.{dataset_name}",
                    config_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if not hasattr(module, 'config'):
                    self.logger.warning(f"No config found in {config_file}")
                    continue
                    
                # Add to configs
                dataset_configs[dataset_name] = module.config
                self.logger.info(f"Loaded config for dataset: {dataset_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading config for {dataset_name}: {str(e)}")
                continue
                
        return dataset_configs
        
    def get_config(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset configuration or None if not found
        """
        return self.configs.get(dataset_name)
        
    def load_dataset(self, dataset_name: str) -> Optional[Any]:
        """Load a dataset instance.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Dataset instance or None if loading fails
        """
        config = self.get_config(dataset_name)
        if not config:
            self.logger.error(f"No configuration found for dataset: {dataset_name}")
            return None
            
        try:
            # Adjust data_root path if needed
            dataset_cfg = config.get('train_dataset', {})
            if 'args' in dataset_cfg and 'data_root' in dataset_cfg['args']:
                if not os.path.isabs(dataset_cfg['args']['data_root']):
                    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
                    dataset_cfg['args']['data_root'] = os.path.join(repo_root, dataset_cfg['args']['data_root'])
            
            # Import the dataset builder
            import utils.builders
            dataset = utils.builders.build_from_config(dataset_cfg)
            self.logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            return None 