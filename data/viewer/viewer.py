"""Dataset viewer module for visualizing datasets."""
from typing import Optional
import os
from pathlib import Path
import dash
from data.viewer.layout.app import create_app_layout
from data.viewer.callbacks import registry
from data.viewer.backend import ViewerBackend

import logging


class DatasetViewer:
    """Dataset viewer class for visualization of datasets."""

    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Initialize the dataset viewer.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
        """
        # Setup logging
        self._setup_logging(log_level, log_file)
        self.logger = logging.getLogger(__name__)

        # Print debug information to help with path issues
        self.logger.info(f"Current working directory: {os.getcwd()}")
        self.logger.info(f"Repository root (estimated): {os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))}")

        # Initialize backend
        self.backend = ViewerBackend()

        # Get available datasets from the backend
        self.available_datasets = self.backend.get_available_datasets()
        self.logger.info(f"Found {len(self.available_datasets)} available datasets")

        # Dash app setup
        self.app = dash.Dash(
            __name__,
            title="Dataset Viewer",
            suppress_callback_exceptions=True,  # Add this to handle callbacks to components created by other callbacks
            prevent_initial_callbacks='initial_duplicate'  # Handle duplicate callbacks properly
        )

        # Create layout
        self.app.layout = create_app_layout(self.available_datasets)

        # Register callbacks
        self._register_callbacks()

    def _setup_logging(self, log_level: str, log_file: Optional[str]) -> None:
        """Setup logging configuration.

        Args:
            log_level: Logging level
            log_file: Optional path to log file
        """
        # Create logs directory if it doesn't exist
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                *([logging.FileHandler(log_file)] if log_file else [])
            ]
        )

    def _register_callbacks(self):
        """Register all callbacks for the app."""
        # Set the viewer instance in the registry
        registry.viewer = self
        # Register all callbacks with the app
        registry.register_callbacks(self.app)

    def run(self, debug=False, host="0.0.0.0", port=8050):
        """Run the viewer application."""
        self.app.run(debug=debug, host=host, port=port)
