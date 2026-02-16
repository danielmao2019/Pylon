"""Dataset viewer module for visualizing datasets."""

import logging
import os
from pathlib import Path
from typing import Optional

from data.viewer.dataset.app import create_app
from data.viewer.dataset.backend import ViewerBackend


class DatasetViewer:
    """Dataset viewer class for visualization of datasets."""

    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Initialize the dataset viewer.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
        """
        self._setup_logging(log_level, log_file)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Current working directory: {os.getcwd()}")
        self.logger.info(
            "Repository root (estimated): "
            f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))}"
        )

        self.backend = ViewerBackend()
        self.available_datasets = self.backend.get_available_datasets_hierarchical()

        total_datasets = sum(
            len(datasets) for datasets in self.available_datasets.values()
        )
        self.logger.info(
            f"Found {total_datasets} available datasets across {len(self.available_datasets)} categories"
        )

        self.app = create_app(self)

    def _setup_logging(self, log_level: str, log_file: Optional[str]) -> None:
        """Setup logging configuration.

        Args:
            log_level: Logging level
            log_file: Optional path to log file
        """
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(log_file)] if log_file else []),
            ],
        )

    def run(self, debug=False, host="0.0.0.0", port=8050):
        """Run the viewer application."""
        self.app.run(debug=debug, host=host, port=port)
