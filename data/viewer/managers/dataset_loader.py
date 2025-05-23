"""Dataset loader module."""
from typing import Dict, Any, Optional, List
import os
import logging
import importlib.util
from data.viewer.managers.registry import DATASET_GROUPS, get_dataset_type


class DatasetLoader:
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
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            self.config_dirs = {
                '2d_change_detection': os.path.join(repo_root, "configs/common/datasets/change_detection/train"),
                '3d_change_detection': os.path.join(repo_root, "configs/common/datasets/change_detection/train"),
                'point_cloud_registration': os.path.join(repo_root, "configs/common/datasets/point_cloud_registration/train")
            }
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
            if dataset_type not in self.config_dirs:
                self.logger.warning(f"Config directory not found for dataset type: {dataset_type}")
                continue

            config_dir = self.config_dirs[dataset_type]
            for dataset_name in DATASET_GROUPS.get(dataset_type, []):
                config_file = os.path.join(config_dir, f"{dataset_name}.py")

                try:
                    if not os.path.isfile(config_file):
                        self.logger.warning(f"Dataset config file not found: {config_file}")
                        continue

                    # Import the config
                    spec = importlib.util.spec_from_file_location(
                        f"configs.common.datasets.{dataset_type}.train.{dataset_name}",
                        config_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if not hasattr(module, 'data_cfg'):
                        self.logger.warning(f"No config found in {config_file}")
                        continue

                    # Add to configs with dataset type prefix
                    config_key = f"{dataset_type}/{dataset_name}"
                    dataset_configs[config_key] = module.data_cfg
                    self.logger.info(f"Loaded config for dataset: {config_key}")

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

        # Adjust data_root path if needed
        dataset_cfg = config.get('train_dataset', {})
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        if 'args' in dataset_cfg and 'data_root' in dataset_cfg['args']:
            if not os.path.isabs(dataset_cfg['args']['data_root']):
                dataset_cfg['args']['data_root'] = os.path.join(repo_root, dataset_cfg['args']['data_root'])
        if 'args' in dataset_cfg and 'gt_transforms_filepath' in dataset_cfg['args']:
            if not os.path.isabs(dataset_cfg['args']['gt_transforms_filepath']):
                dataset_cfg['args']['gt_transforms_filepath'] = os.path.join(repo_root, dataset_cfg['args']['gt_transforms_filepath'])

        # Import the dataset builder
        import utils.builders
        dataset = utils.builders.build_from_config(dataset_cfg)
        self.logger.info(f"Successfully loaded dataset: {dataset_name}")
        return dataset
