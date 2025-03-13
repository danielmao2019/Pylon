"""Dataset viewer module for visualizing datasets."""
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import traceback
import os

# Import project modules
import data
import utils

# Import viewer sub-modules
from data.viewer.utils.dataset_utils import get_available_datasets, format_value, is_3d_dataset
from data.viewer.ui.display import display_2d_datapoint, display_3d_datapoint
from data.viewer.ui.controls import (
    create_dataset_selector, 
    create_reload_button, 
    create_navigation_controls, 
    create_3d_controls,
    create_dataset_info_display
)
from data.viewer.ui.transforms import create_transforms_section


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

    def _load_dataset(self, dataset_name):
        """Load a selected dataset and reset the datapoint slider."""
        if dataset_name is None:
            return (
                {}, 0, 0, 0, {},
                html.Div("No dataset selected."),
                create_dataset_info_display(),
                create_transforms_section(),
                False
            )

        # Special case for the test dataset
        if dataset_name == 'test_dataset':
            dataset_info = {'name': 'test_dataset', 'length': 10, 'class_labels': {}}
            return (
                dataset_info,
                0, 9, 0, {i: str(i) for i in range(0, 10, 2)},
                html.Div([
                    html.H2("Test Dataset Viewer", style={'text-align': 'center'}),
                    html.P("This is a placeholder UI for testing when no real datasets are available."),
                    html.P("Please ensure your config directories are correctly set up and accessible."),
                    html.Div(style={'background-color': '#f8f9fa', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'}, children=[
                        html.H3("Troubleshooting Tips:"),
                        html.Ul([
                            html.Li("Check that the repository structure is correct"),
                            html.Li("Verify that dataset configurations exist in your repository"),
                            html.Li("Make sure dataset config files have the expected format"),
                            html.Li("Run the script from the repository root instead of the data/datasets directory")
                        ])
                    ])
                ]),
                create_dataset_info_display(dataset_info),
                create_transforms_section(),
                False
            )

        try:
            dataset = self.datasets.get(dataset_name)
            if dataset is None:
                return (
                    {}, 0, 0, 0, {},
                    html.Div(f"Failed to load dataset: {dataset_name}. Dataset configuration might be invalid."),
                    create_dataset_info_display(),
                    create_transforms_section(),
                    False
                )

            # Create dataset info for the store
            dataset_length = len(dataset)

            # Create slider marks at appropriate intervals
            marks = {}
            if dataset_length <= 10:
                # If less than 10 items, mark each one
                marks = {i: str(i) for i in range(dataset_length)}
            else:
                # Otherwise, create marks at regular intervals
                step = max(1, dataset_length // 10)
                marks = {i: str(i) for i in range(0, dataset_length, step)}
                # Always include the last index
                marks[dataset_length - 1] = str(dataset_length - 1)

            # Get class labels if available
            class_labels = {}
            try:
                # Try to access class labels from the dataset if they exist
                if hasattr(dataset, 'class_names'):
                    class_labels = {i: name for i, name in enumerate(dataset.class_names)}
                # For some datasets, class_names might be nested
                elif hasattr(dataset, 'labels') and hasattr(dataset.labels, 'class_names'):
                    class_labels = {i: name for i, name in enumerate(dataset.labels.class_names)}
            except Exception as e:
                print(f"Warning: Could not get class labels from dataset: {e}")

            # Determine if this is a 3D dataset based on class name
            is_3d = is_3d_dataset(dataset)

            # Create dataset info for the data store
            dataset_info = {
                'name': dataset_name,
                'length': dataset_length,
                'class_labels': class_labels
            }

            # Get first datapoint for display
            initial_message = html.Div(f"Dataset '{dataset_name}' loaded successfully with {dataset_length} datapoints. Use the slider to navigate.")

            return (
                dataset_info,
                0,                   # min slider value
                dataset_length - 1,  # max slider value
                0,                   # initial slider value
                marks,               # slider marks
                initial_message,
                create_dataset_info_display(dataset_info),
                create_transforms_section(self.available_datasets.get(dataset_name)),
                is_3d                # is 3D dataset flag
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            return (
                {}, 0, 0, 0, {},
                html.Div([
                    html.H3(f"Error Loading Dataset: {str(e)}", style={'color': 'red'}),
                    html.Pre(error_traceback, style={
                        'background-color': '#ffeeee',
                        'padding': '10px',
                        'border-radius': '5px',
                        'max-height': '300px',
                        'overflow-y': 'auto'
                    })
                ]),
                create_dataset_info_display(),
                create_transforms_section(),
                False
            )

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
        # Callback to load dataset
        self.app.callback(
            [
                Output('dataset-info', 'data'),
                Output('datapoint-index-slider', 'min'),
                Output('datapoint-index-slider', 'max'),
                Output('datapoint-index-slider', 'value'),
                Output('datapoint-index-slider', 'marks'),
                Output('datapoint-display', 'children', allow_duplicate=True),
                Output('dataset-info-display', 'children'),
                Output('transforms-section', 'children'),
                Output('is-3d-dataset', 'data', allow_duplicate=True)
            ],
            [Input('dataset-dropdown', 'value')],
            prevent_initial_call=True
        )(self._load_dataset)

        # Callback to update datapoint display
        self.app.callback(
            [
                Output('datapoint-display', 'children'),
                Output('is-3d-dataset', 'data')
            ],
            [
                Input('dataset-info', 'data'),
                Input('datapoint-index-slider', 'value'),
                Input('point-size-slider', 'value'),
                Input('point-opacity-slider', 'value')
            ],
            [State('is-3d-dataset', 'data')]
        )(self._update_datapoint)

        # Callback to apply transforms
        self.app.callback(
            Output('datapoint-display', 'children', allow_duplicate=True),
            [Input({'type': 'transform-checkbox', 'index': ALL}, 'value')],
            [
                State('dataset-info', 'data'),
                State('datapoint-index-slider', 'value')
            ],
            prevent_initial_call=True
        )(self._apply_transforms)

        # Callback for prev/next buttons
        self.app.callback(
            Output('datapoint-index-slider', 'value'),
            [
                Input('prev-btn', 'n_clicks'),
                Input('next-btn', 'n_clicks'),
            ],
            [
                State('datapoint-index-slider', 'value'),
                State('datapoint-index-slider', 'min'),
                State('datapoint-index-slider', 'max'),
            ],
            prevent_initial_call=True
        )(self._update_index_from_buttons)

        # Callback to update index display
        self.app.callback(
            Output('current-index-display', 'children'),
            [Input('datapoint-index-slider', 'value')],
            [State('dataset-info', 'data')]
        )(self._update_index_display)

        # Callback to update view controls
        self.app.callback(
            Output('view-controls', 'style'),
            [Input('is-3d-dataset', 'data')]
        )(self._update_view_controls)

        # Callback to reload datasets
        self.app.callback(
            Output('dataset-dropdown', 'options'),
            [Input('reload-button', 'n_clicks')],
            prevent_initial_call=True
        )(self._reload_datasets)

    def _update_datapoint(self, dataset_info, datapoint_idx, point_size, point_opacity, is_3d_prev):
        """
        Update the displayed datapoint based on the slider value.
        Also handles 3D point cloud visualization settings.
        """
        if dataset_info is None or dataset_info == {}:
            return html.Div("No dataset loaded."), False

        try:
            dataset_name = dataset_info.get('name', 'unknown')
            dataset = self.datasets.get(dataset_name)
            if dataset is None:
                return html.Div(f"Dataset '{dataset_name}' not found."), False

            # Get the datapoint
            if datapoint_idx >= len(dataset):
                return html.Div(f"Datapoint index {datapoint_idx} is out of range for dataset of size {len(dataset)}."), is_3d_prev

            datapoint = dataset[datapoint_idx]

            # Determine if this is a 3D dataset based on the class name or data structure
            is_3d = is_3d_dataset(dataset, datapoint)

            # Get class labels if available
            class_labels = dataset_info.get('class_labels', {})

            # Display the datapoint based on its type
            try:
                if is_3d:
                    display = display_3d_datapoint(datapoint, point_size, point_opacity, class_labels)
                else:
                    display = display_2d_datapoint(datapoint)
            except Exception as e:
                error_traceback = traceback.format_exc()
                return html.Div([
                    html.H3(f"Error Loading Datapoint: {str(e)}", style={'color': 'red'}),
                    html.P("Dataset type detection:"),
                    html.Pre(f"Dataset class: {dataset.__class__.__name__}"),
                    html.Pre(f"Is 3D: {is_3d}"),
                    html.P("Datapoint structure:"),
                    html.Pre(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                    html.Pre(f"Labels keys: {list(datapoint['labels'].keys())}"),
                    html.Pre(f"Meta info: {format_value(datapoint.get('meta_info', {}))}"),
                    html.P("Error traceback:"),
                    html.Pre(error_traceback, style={
                        'background-color': '#ffeeee',
                        'padding': '10px',
                        'border-radius': '5px',
                        'max-height': '300px',
                        'overflow-y': 'auto'
                    })
                ]), is_3d

            return display, is_3d

        except Exception as e:
            error_traceback = traceback.format_exc()
            return html.Div([
                html.H3("Error", style={'color': 'red'}),
                html.P(f"Error loading datapoint: {str(e)}")
            ]), is_3d_prev

    def _apply_transforms(self, transform_values, dataset_info, datapoint_idx):
        """Apply the selected transforms to the current datapoint."""
        if not dataset_info or 'name' not in dataset_info:
            raise PreventUpdate

        dataset_name = dataset_info['name']
        dataset = self.datasets.get(dataset_name)

        if dataset is None or datapoint_idx >= len(dataset):
            raise PreventUpdate

        # Get original datapoint
        datapoint = dataset[datapoint_idx]

        # Get transforms configuration
        selected_indices = [indices[0] for indices in transform_values if indices]

        if not selected_indices:
            # If no transforms selected, just return the datapoint as is
            try:
                is_3d = is_3d_dataset(dataset, datapoint)
                class_labels = dataset_info.get('class_labels', {})
                if is_3d:
                    return display_3d_datapoint(datapoint, class_names=class_labels)
                else:
                    return display_2d_datapoint(datapoint)
            except Exception as e:
                return html.Div([
                    html.H3("Error Displaying Datapoint", style={'color': 'red'}),
                    html.P(f"Error: {str(e)}")
                ])

        try:
            # Get the dataset configuration
            dataset_cfg = self.available_datasets[dataset_name].get('train_dataset', {})
            transforms_cfg = dataset_cfg.get('args', {}).get('transforms_cfg')

            if not transforms_cfg or 'args' not in transforms_cfg or 'transforms' not in transforms_cfg['args']:
                raise PreventUpdate

            # Get all available transforms
            all_transforms = transforms_cfg['args']['transforms']

            # Filter transforms by selected indices
            selected_transforms = [all_transforms[i] for i in selected_indices if i < len(all_transforms)]

            # Create a transforms configuration with only selected transforms
            filtered_transforms_cfg = {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': selected_transforms
                }
            }

            # Build the transforms pipeline
            transforms_pipeline = utils.builders.build_from_config(filtered_transforms_cfg)

            # Apply transforms to the datapoint
            transformed_datapoint = transforms_pipeline(datapoint)

            # Display the transformed datapoint
            is_3d = is_3d_dataset(dataset, transformed_datapoint)
            class_labels = dataset_info.get('class_labels', {})
            if is_3d:
                return display_3d_datapoint(transformed_datapoint, class_names=class_labels)
            else:
                return display_2d_datapoint(transformed_datapoint)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return html.Div([
                html.H3("Error Applying Transforms", style={'color': 'red'}),
                html.P(f"Error: {str(e)}"),
                html.Pre(error_traceback, style={
                    'background-color': '#ffeeee',
                    'padding': '10px',
                    'border-radius': '5px',
                    'max-height': '300px',
                    'overflow-y': 'auto'
                })
            ])

    def _update_index_from_buttons(self, prev_clicks, next_clicks, current_idx, min_idx, max_idx):
        """Update the slider index when prev/next buttons are clicked."""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'prev-btn' and current_idx > min_idx:
            return current_idx - 1
        elif trigger_id == 'next-btn' and current_idx < max_idx:
            return current_idx + 1

        return current_idx

    def _update_index_display(self, index, dataset_info):
        """Update the displayed index value."""
        if dataset_info and 'length' in dataset_info:
            return f"Index: {index} / {dataset_info['length'] - 1}"
        return f"Index: {index}"

    def _update_view_controls(self, is_3d):
        """Update the visibility of 3D view controls based on dataset type."""
        if is_3d:
            return {'display': 'block', 'margin-top': '20px'}
        else:
            return {'display': 'none'}

    def _reload_datasets(self, n_clicks):
        """Reload the list of available datasets."""
        if n_clicks is None:
            raise PreventUpdate

        # Refresh the available datasets
        self.available_datasets = get_available_datasets()
        self._load_available_datasets()

        # Return updated options
        return [{'label': name, 'value': name} for name in sorted(self.available_datasets.keys())]

    def run(self, debug=True, host='0.0.0.0', port=8050):
        """Run the Dash application."""
        self.app.run_server(debug=debug, host=host, port=port)
