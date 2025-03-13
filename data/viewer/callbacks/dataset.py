"""Dataset-related callbacks for the viewer."""
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import html
import traceback
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.utils.dataset_utils import get_available_datasets, is_3d_dataset
from data.viewer.layout.controls.dataset import create_dataset_info_display
from data.viewer.layout.controls.transforms import create_transforms_section

def register_dataset_callbacks(app, viewer):
    """Register callbacks related to dataset operations."""
    
    @app.callback(
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
    )
    def load_dataset(dataset_name):
        """Load a selected dataset and reset the datapoint slider."""
        if dataset_name is None:
            viewer.state.reset()
            return (
                {}, 0, 0, 0, {},
                html.Div("No dataset selected."),
                create_dataset_info_display(),
                create_transforms_section(),
                False
            )

        # Special case for the test dataset
        if dataset_name == 'test_dataset':
            viewer.state.update_dataset_info(
                name='test_dataset',
                length=10,
                class_labels={},
                is_3d=False,
                available_transforms=[]
            )
            return (
                viewer.state.get_state()['dataset_info'],
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
                create_dataset_info_display(viewer.state.get_state()['dataset_info']),
                create_transforms_section(),
                False
            )

        try:
            dataset = viewer.datasets.get(dataset_name)
            if dataset is None:
                viewer.state.reset()
                return (
                    {}, 0, 0, 0, {},
                    html.Div(f"Failed to load dataset: {dataset_name}. Dataset configuration might be invalid."),
                    create_dataset_info_display(),
                    create_transforms_section(),
                    False
                )

            # Get dataset information
            dataset_length = len(dataset)
            is_3d = is_3d_dataset(dataset)

            # Get class labels if available
            class_labels = {}
            try:
                if hasattr(dataset, 'class_names'):
                    class_labels = {i: name for i, name in enumerate(dataset.class_names)}
                elif hasattr(dataset, 'labels') and hasattr(dataset.labels, 'class_names'):
                    class_labels = {i: name for i, name in enumerate(dataset.labels.class_names)}
            except Exception as e:
                print(f"Warning: Could not get class labels from dataset: {e}")

            # Get available transforms from dataset config
            available_transforms = []
            dataset_cfg = viewer.available_datasets.get(dataset_name, {}).get('train_dataset', {})
            if 'args' in dataset_cfg and 'transforms_cfg' in dataset_cfg['args']:
                transforms_cfg = dataset_cfg['args']['transforms_cfg']
                if 'args' in transforms_cfg and 'transforms' in transforms_cfg['args']:
                    available_transforms = [t.get('class', '') for t in transforms_cfg['args']['transforms']]

            # Update state
            viewer.state.update_dataset_info(
                name=dataset_name,
                length=dataset_length,
                class_labels=class_labels,
                is_3d=is_3d,
                available_transforms=available_transforms
            )

            # Create slider marks
            marks = {}
            if dataset_length <= 10:
                marks = {i: str(i) for i in range(dataset_length)}
            else:
                step = max(1, dataset_length // 10)
                marks = {i: str(i) for i in range(0, dataset_length, step)}
                marks[dataset_length - 1] = str(dataset_length - 1)

            # Get initial message
            initial_message = html.Div(f"Dataset '{dataset_name}' loaded successfully with {dataset_length} datapoints. Use the slider to navigate.")

            return (
                viewer.state.get_state()['dataset_info'],
                0,                   # min slider value
                dataset_length - 1,  # max slider value
                0,                   # initial slider value
                marks,               # slider marks
                initial_message,
                create_dataset_info_display(viewer.state.get_state()['dataset_info']),
                create_transforms_section(available_transforms),
                is_3d                # is 3D dataset flag
            )
        except Exception as e:
            error_traceback = traceback.format_exc()
            viewer.state.reset()
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

    @app.callback(
        Output('dataset-dropdown', 'options'),
        [Input('reload-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def reload_datasets(n_clicks):
        """Reload available datasets."""
        if n_clicks is None:
            raise PreventUpdate
            
        # Get updated list of datasets
        viewer.available_datasets = get_available_datasets()
        
        # Create options for the dropdown
        options = [
            {'label': name, 'value': name}
            for name in viewer.available_datasets.keys()
        ]
        
        return options 