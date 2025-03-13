"""Dataset-related callbacks for the viewer."""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import html
import traceback
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.display.dataset import create_dataset_info_display
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.callbacks.registry import callback


@callback(
    outputs=[
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
    inputs=[Input('dataset-dropdown', 'value')],
    group="dataset"
)
def load_dataset(dataset_name):
    """Load a selected dataset and reset the datapoint slider."""
    def create_error_display(message, error_trace=None):
        """Helper to create error display."""
        error_content = [html.H3(f"Error Loading Dataset: {message}", style={'color': 'red'})]
        if error_trace:
            error_content.append(html.Pre(error_trace, style={
                'background-color': '#ffeeee',
                'padding': '10px',
                'border-radius': '5px',
                'max-height': '300px',
                'overflow-y': 'auto'
            }))
        return html.Div(error_content)

    def get_default_outputs(error_display):
        """Helper to get default outputs when loading fails."""
        viewer.state.reset()
        return (
            {}, 0, 0, 0, {},
            error_display,
            create_dataset_info_display(),
            create_transforms_section(),
            False
        )

    if dataset_name is None:
        return get_default_outputs(html.Div("No dataset selected."))

    try:
        # Load dataset using dataset manager
        success, message, dataset_info = viewer.dataset_manager.load_dataset(dataset_name)

        if not success:
            return get_default_outputs(create_error_display(message, traceback.format_exc()))

        # Update state with dataset info
        viewer.state.update_dataset_info(
            name=dataset_info['name'],
            length=dataset_info['length'],
            class_labels=dataset_info['class_labels'],
            is_3d=dataset_info['is_3d'],
            available_transforms=dataset_info['available_transforms']
        )

        # Create slider marks
        marks = {}
        if dataset_info['length'] <= 10:
            marks = {i: str(i) for i in range(dataset_info['length'])}
        else:
            step = max(1, dataset_info['length'] // 10)
            marks = {i: str(i) for i in range(0, dataset_info['length'], step)}
            marks[dataset_info['length'] - 1] = str(dataset_info['length'] - 1)

        # Get initial message
        initial_message = html.Div(f"Dataset '{dataset_name}' loaded successfully with {dataset_info['length']} datapoints. Use the slider to navigate.")

        return (
            viewer.state.get_state()['dataset_info'],
            0,                   # min slider value
            dataset_info['length'] - 1,  # max slider value
            0,                   # initial slider value
            marks,               # slider marks
            initial_message,
            create_dataset_info_display(viewer.state.get_state()['dataset_info']),
            create_transforms_section(dataset_info['available_transforms']),
            dataset_info['is_3d']  # is 3D dataset flag
        )

    except Exception as e:
        return get_default_outputs(create_error_display(str(e), traceback.format_exc()))


@callback(
    outputs=Output('dataset-dropdown', 'options'),
    inputs=[Input('reload-button', 'n_clicks')],
    group="dataset"
)
def reload_datasets(n_clicks):
    """Reload available datasets."""
    if n_clicks is None:
        raise PreventUpdate

    # Get updated list of datasets
    viewer.dataset_manager._load_available_datasets()
    available_datasets = viewer.dataset_manager._configs

    # Create options for the dropdown
    options = [
        {'label': name, 'value': name}
        for name in available_datasets
    ]

    return options
