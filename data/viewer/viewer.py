"""Dataset viewer module for visualizing datasets."""
import dash
import traceback
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Import viewer sub-modules
from data.viewer.layout.controls.dataset import get_available_datasets
from data.viewer.layout import create_app_layout

# Import callback modules
from data.viewer.callbacks.dataset import register_dataset_callbacks
from data.viewer.callbacks.display import register_display_callbacks
from data.viewer.callbacks.navigation import register_navigation_callbacks
from data.viewer.callbacks.transforms import register_transform_callbacks

# Import state management
from data.viewer.states import ViewerState

# Other project imports - required for functionality
import utils.builders


class ViewerError(Exception):
    """Base exception for viewer-related errors."""
    pass


class DatasetLoadError(ViewerError):
    """Exception raised when dataset loading fails."""
    pass


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

        # Get all available datasets
        self.available_datasets = get_available_datasets()
        self.logger.info(f"Found {len(self.available_datasets)} available datasets")

        # Store for available datasets
        self.datasets: Dict[str, Any] = {}

        # Initialize state management
        self.state = ViewerState()

        # Load datasets
        self._load_available_datasets()

        # Dash app setup
        self.app = dash.Dash(
            __name__,
            title="Dataset Viewer",
            suppress_callback_exceptions=True  # Add this to handle callbacks to components created by other callbacks
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

    def _load_available_datasets(self) -> None:
        """Load available datasets from configuration files."""
        for name, config in self.available_datasets.items():
            try:
                # Adjust data_root path to be relative to the repository root if needed
                dataset_cfg = config.get('train_dataset', {})
                if 'args' in dataset_cfg and 'data_root' in dataset_cfg['args'] and not os.path.isabs(dataset_cfg['args']['data_root']):
                    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
                    dataset_cfg['args']['data_root'] = os.path.join(repo_root, dataset_cfg['args']['data_root'])

                # Build dataset
                dataset = utils.builders.build_from_config(dataset_cfg)
                self.datasets[name] = dataset
                self.logger.info(f"Loaded dataset: {name}")
            except Exception as e:
                self.logger.error(f"Error loading dataset: {name}: {e}")
                self.logger.error(traceback.format_exc())
                raise DatasetLoadError(f"Failed to load dataset {name}: {str(e)}")

    def _register_callbacks(self):
        """Register all callbacks for the app."""
        # Register callbacks from each module
        register_dataset_callbacks(self.app, self)
        register_display_callbacks(self.app, self)
        register_navigation_callbacks(self.app, self)
        register_transform_callbacks(self.app, self)

    def run(self, debug=False, host="0.0.0.0", port=8050):
        """Run the viewer application."""
        self.app.run_server(debug=debug, host=host, port=port)
