"""Dataset-related callbacks for the viewer."""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import html
import logging
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.display.dataset import create_dataset_info_display
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.callbacks.registry import callback

logger = logging.getLogger(__name__)

@callback(
    outputs=[
        Output('dataset-info', 'data'),
        Output('datapoint-index-slider', 'min'),
        Output('datapoint-index-slider', 'max'),
        Output('datapoint-index-slider', 'value'),
        Output('datapoint-index-slider', 'marks'),
        Output('datapoint-display', 'children', allow_duplicate=True),
        Output('dataset-info-display', 'children'),
        Output('transforms-section', 'children')
    ],
    inputs=[Input('dataset-dropdown', 'value')],
    group="dataset"
)
def load_dataset(dataset_name):
    """Load a selected dataset and reset the datapoint slider."""
    logger.info(f"Dataset loading callback triggered with dataset: {dataset_name}")
    
    if dataset_name is None:
        logger.info("No dataset selected")
        viewer.state.reset()
        return (
            {}, 0, 0, 0, {},
            html.Div("No dataset selected."),
            create_dataset_info_display(),
            create_transforms_section(),
        )

    # Load dataset using dataset manager
    logger.info(f"Attempting to load dataset: {dataset_name}")
    success, message, dataset_info = viewer.dataset_manager.load_dataset(dataset_name)
    logger.info(f"Dataset load result - Success: {success}, Message: {message}")

    if not success:
        logger.error(f"Failed to load dataset: {message}")
        viewer.state.reset()
        return (
            {}, 0, 0, 0, {},
            html.Div(f"Error Loading Dataset: {message}"),
            create_dataset_info_display(),
            create_transforms_section(),
        )

    # Update state with dataset info
    logger.info(f"Updating state with dataset info: {dataset_info}")
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
    logger.info("Dataset loaded successfully, returning updated UI components")

    return (
        viewer.state.get_state()['dataset_info'],
        0,                   # min slider value
        dataset_info['length'] - 1,  # max slider value
        0,                   # initial slider value
        marks,               # slider marks
        initial_message,
        create_dataset_info_display(viewer.state.get_state()['dataset_info']),
        create_transforms_section(dataset_info['available_transforms']),
    )


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
