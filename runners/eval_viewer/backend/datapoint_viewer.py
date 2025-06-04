"""Backend functionality for datapoint viewing in eval viewer."""
from typing import Dict, Tuple
import json
import os
import logging
from data.viewer.managers.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


def resolve_dataset_from_log_dir(log_dir: str) -> Tuple[str, Dict]:
    """Resolve the dataset class and configuration from a log directory.

    Args:
        log_dir: Path to the log directory

    Returns:
        Tuple containing:
            - dataset_class_name: Name of the dataset class (e.g., 'LevirCdDataset')
            - dataset_config: Dictionary containing dataset configuration

    Raises:
        FileNotFoundError: If config.json doesn't exist
        KeyError: If required dataset config is missing
    """
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract dataset info from val_dataset config
    dataset_config = config['val_dataset']
    dataset_class = dataset_config['class']

    return dataset_class, dataset_config


class DatapointViewer:
    """Component for displaying individual datapoint information in the eval viewer."""

    def __init__(self, dataset_manager: DatasetManager):
        """Initialize the datapoint viewer.

        Args:
            dataset_manager: DatasetManager instance from data.viewer for loading datapoints
        """
        self.dataset_manager = dataset_manager
        self.dataset_configs = {}  # Cache of dataset configs per log directory
        self.current_dataset = None
        self.current_datapoint_idx = None

    def get_dataset_config(self, log_dir: str) -> Tuple[str, Dict]:
        """Get the dataset configuration for a log directory, using cache if available.

        Args:
            log_dir: Path to log directory

        Returns:
            Tuple containing:
                - dataset_class_name: Name of the dataset class
                - dataset_config: Dictionary containing dataset configuration
        """
        if log_dir not in self.dataset_configs:
            self.dataset_configs[log_dir] = resolve_dataset_from_log_dir(log_dir)
        return self.dataset_configs[log_dir]

    def load_datapoint(self, log_dir: str, datapoint_idx: int) -> Dict:
        """Load a specific datapoint from the dataset.

        Args:
            log_dir: Path to log directory
            datapoint_idx: Index of the datapoint to load

        Returns:
            Dict containing the datapoint data
        """
        dataset_class, dataset_config = self.get_dataset_config(log_dir)

        # Get dataset name from class (e.g., 'LevirCdDataset' -> 'levir_cd')
        dataset_name = dataset_class.lower().replace('dataset', '')

        # Load dataset using dataset manager
        dataset = self.dataset_manager.get_dataset(dataset_name)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_name} not found in dataset manager")

        # Load the specific datapoint
        datapoint = dataset[datapoint_idx]

        # Update current state
        self.current_dataset = dataset_name
        self.current_datapoint_idx = datapoint_idx

        return datapoint
