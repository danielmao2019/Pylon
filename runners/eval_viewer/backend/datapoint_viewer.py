"""Backend functionality for datapoint viewing in eval viewer."""
from typing import Dict
import logging
from data.viewer.managers.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class DatapointViewer:
    """Component for displaying individual datapoint information in the eval viewer."""

    def __init__(self, dataset_manager: DatasetManager, dataset_class: str, dataset_type: DatasetType):
        """Initialize the datapoint viewer.

        Args:
            dataset_manager: DatasetManager instance from data.viewer for loading datapoints
        """
        self.dataset_manager = dataset_manager
        self.dataset_class = dataset_class
        self.dataset_type = dataset_type
        self.dataset = dataset_manager.get_dataset(dataset_class)
        self.current_datapoint_idx = None

    def load_datapoint(self, datapoint_idx: int) -> Dict:
        """Load a specific datapoint from the dataset.

        Args:
            datapoint_idx: Index of the datapoint to load

        Returns:
            Dict containing the datapoint data
        """
        datapoint = self.dataset[datapoint_idx]
        self.current_datapoint_idx = datapoint_idx
        return datapoint
