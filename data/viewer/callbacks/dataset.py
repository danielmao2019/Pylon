"""Dataset-related callbacks for the viewer."""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import logging
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.display.dataset import create_dataset_info_display
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.callbacks.registry import callback, registry

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
        registry.viewer.state.reset()
        return (
            {},  # dataset-info
            0,   # min
            0,   # max
            [0], # value
            {},  # marks
            html.Div("No dataset selected."),  # datapoint-display
            create_dataset_info_display(),  # dataset-info-display
            create_transforms_section(),  # transforms-section
        )

    # Load dataset using dataset manager
    logger.info(f"Attempting to load dataset: {dataset_name}")
    dataset_info = registry.viewer.dataset_manager.load_dataset(dataset_name)

    # Update state with dataset info
    logger.info(f"Updating state with dataset info: {dataset_info}")
    registry.viewer.state.update_dataset_info(
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
        registry.viewer.state.get_state()['dataset_info'],  # dataset-info
        0,                   # min
        dataset_info['length'] - 1,  # max
        [0],                # value
        marks,              # marks
        initial_message,    # datapoint-display
        create_dataset_info_display(registry.viewer.state.get_state()['dataset_info']),  # dataset-info-display
        create_transforms_section(dataset_info['available_transforms']),  # transforms-section
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
    registry.viewer.dataset_manager._load_available_datasets()
    available_datasets = registry.viewer.dataset_manager._configs

    # Create options for the dropdown
    options = [
        {'label': name, 'value': name}
        for name in available_datasets
    ]

    return options
