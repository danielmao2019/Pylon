"""Dataset loader module."""
from typing import Dict, Any, Optional, List
import os
import logging
import importlib.util
from data.viewer.managers.registry import DATASET_GROUPS, CONFIG_DIRS, get_dataset_type


class DatasetBuilder:
    """Handles loading and configuration of datasets."""

    def __init__(self, config_dir: Optional[str] = None, dataset_types: Optional[List[str]] = None):
        """Initialize the dataset loader.

        Args:
            config_dir: Optional directory containing dataset configurations
            dataset_types: Optional list of dataset types to load (e.g., ['2d_change_detection', 'point_cloud_registration'])
        """
        self.logger = logging.getLogger(__name__)

        # If no config_dir is provided, use the default locations
        if config_dir is None:
            self.config_dirs = CONFIG_DIRS
        else:
            self.config_dirs = {'default': config_dir}

        # Default to loading all dataset types if none specified
        self.dataset_types = dataset_types or list(DATASET_GROUPS.keys())
        self.configs = self._load_dataset_configs()

    def _load_dataset_configs(self) -> Dict[str, Any]:
        """Load all available dataset configurations.

        Returns:
            Dictionary mapping dataset names to their configurations
        """
        dataset_configs = {}

        for dataset_type in self.dataset_types:
            assert dataset_type in self.config_dirs, f"Config directory not found for dataset type: {dataset_type}"

            config_dir = self.config_dirs[dataset_type]
            for dataset_name in DATASET_GROUPS[dataset_type]:
                config_file = os.path.join(config_dir, f"{dataset_name}_data_cfg.py")
                assert os.path.isfile(config_file), f"Dataset config file not found: {config_file}"

                # Import the config
                spec = importlib.util.spec_from_file_location("config_file", config_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                assert hasattr(module, 'data_cfg'), f"No config found in {config_file}"

                # Add to configs with dataset type prefix
                config_key = f"{dataset_type}/{dataset_name}"
                dataset_configs[config_key] = module.data_cfg
                self.logger.info(f"Loaded config for dataset: {config_key}")

        return dataset_configs

    def get_config(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset configuration or None if not found
        """
        dataset_type = get_dataset_type(dataset_name)
        config_key = f"{dataset_type}/{dataset_name}"
        assert config_key in self.configs, f"Config not found for dataset: {config_key}"
        return self.configs[config_key]

    def build_dataset(self, dataset_name: str) -> Optional[Any]:
        """Load a dataset instance.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Dataset instance or None if loading fails
        """
        config = self.get_config(dataset_name)

        dataset_cfg = config.get('train_dataset', {})

        # Handle relative paths in dataset config
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        path_keys = ['data_root', 'gt_transforms_filepath']
        if 'args' in dataset_cfg:
            for key in path_keys:
                print(f"Adjusting path for {key}")
                if key in dataset_cfg['args'] and not os.path.isabs(dataset_cfg['args'][key]):
                    dataset_cfg['args'][key] = os.path.normpath(os.path.join(repo_root, dataset_cfg['args'][key]))

        # Import the dataset builder
        import utils.builders
        dataset = utils.builders.build_from_config(dataset_cfg)
        self.logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset
