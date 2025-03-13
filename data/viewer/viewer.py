"""Dataset viewer module for visualizing datasets."""
import dash
from dash import dcc, html
import traceback
import os

# Import viewer sub-modules
from data.viewer.utils.dataset_utils import get_available_datasets, format_value, is_3d_dataset
from data.viewer.layout.controls.dataset import create_dataset_selector, create_dataset_info_display
from data.viewer.layout.controls.navigation import create_navigation_controls, create_reload_button
from data.viewer.layout.controls.controls_3d import create_3d_controls
from data.viewer.layout.controls.transforms import create_transforms_section

# Import callback modules
from data.viewer.callbacks.dataset import register_dataset_callbacks
from data.viewer.callbacks.display import register_display_callbacks
from data.viewer.callbacks.navigation import register_navigation_callbacks
from data.viewer.callbacks.transforms import register_transform_callbacks

# Import state management
from data.viewer.states import ViewerState

# Other project imports - required for functionality
import data
import utils.builders


class DatasetViewer:
    """Dataset viewer class for visualization of datasets."""

    def __init__(self):
        """Initialize the dataset viewer."""
        # Print debug information to help with path issues
        print(f"Current working directory: {os.getcwd()}")
        print(f"Repository root (estimated): {os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))}")

        # Get all available datasets
        self.available_datasets = get_available_datasets()

        # Store for available datasets
        self.datasets = {}

        # Initialize state management
        self.state = ViewerState()

        # Load datasets
        self._load_available_datasets()

        # Add a test dataset if no datasets are available
        if not self.available_datasets:
            print("Warning: No datasets were found. Adding a fallback dataset for testing.")
            self.available_datasets = {
                "test_dataset": {
                    "train_dataset": {
                        "class": "TestDataset",
                        "args": {
                            "data_root": "./",
                        }
                    }
                }
            }
            # Add the test dataset to available datasets
            self.datasets["test_dataset"] = None

        # Dash app setup
        self.app = dash.Dash(
            __name__,
            title="Dataset Viewer",
            suppress_callback_exceptions=True  # Add this to handle callbacks to components created by other callbacks
        )

        # Create layout
        self.app.layout = self._create_app_layout()

        # Register callbacks
        self._register_callbacks()

    def _load_available_datasets(self):
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
                print(f"Loaded dataset: {name}")
            except Exception as e:
                print(f"Error loading dataset: {name}: {e}")

    def _create_app_layout(self):
        """Create the main application layout."""
        # Initialize Dash components
        layout = html.Div([
            # Hidden stores for keeping track of state
            dcc.Store(id='dataset-info', data={}),
            dcc.Store(id='is-3d-dataset', data=False),

            # Header
            html.Div([
                html.H1("Dataset Viewer", style={'text-align': 'center', 'margin-bottom': '20px'}),

                # Dataset selector and reload button
                html.Div([
                    create_dataset_selector(self.available_datasets),
                    create_reload_button()
                ], style={'display': 'flex', 'align-items': 'flex-end'}),

                # Navigation controls
                create_navigation_controls()
            ], style={'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px', 'margin-bottom': '20px'}),

            # Main content area
            html.Div([
                # Left sidebar with controls and info
                html.Div([
                    # Dataset info section
                    html.Div(id='dataset-info-display', style={'margin-bottom': '20px'}),
                    
                    # Transforms section
                    html.Div(id='transforms-section', style={'margin-bottom': '20px'}),

                    # 3D View Controls - initially hidden, shown only for 3D datasets
                    create_3d_controls(visible=False)
                ], style={'width': '25%', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px'}),

                # Right main display area
                html.Div([
                    html.Div(id='datapoint-display', style={'padding': '10px'})
                ], style={'width': '75%', 'padding': '20px', 'background-color': '#ffffff', 'border-radius': '5px'})
            ], style={'display': 'flex', 'gap': '20px'})
        ])

        return layout

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
