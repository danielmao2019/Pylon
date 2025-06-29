"""Dataset-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.controls.transforms import create_transforms_section
from data.viewer.layout.display.dataset import create_dataset_info_display

import logging
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
def load_dataset(dataset_key: Optional[str]) -> List[Union[Dict[str, Any], int, html.Div]]:
    """Load a selected dataset and reset the datapoint slider."""
    logger.info(f"Dataset loading callback triggered with dataset: {dataset_key}")

    if dataset_key is None:
        logger.info("No dataset selected")
        registry.viewer.backend.current_dataset = None
        return [
            {},  # dataset-info
            0,   # min
            0,   # max
            0,   # value
            {},  # marks
            html.Div("No dataset selected."),  # datapoint-display
            create_dataset_info_display(),  # dataset-info-display
            create_transforms_section(),  # transforms-section
        ]

    # Load dataset using backend
    logger.info(f"Attempting to load dataset: {dataset_key}")
    dataset_info = registry.viewer.backend.load_dataset(dataset_key)

    # Update backend state
    logger.info(f"Updating backend state with dataset info: {dataset_info}")
    registry.viewer.backend.update_state(
        current_dataset=dataset_info['name'],
        current_index=0
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
    initial_message = html.Div(f"Dataset '{dataset_key}' loaded successfully with {dataset_info['length']} datapoints. Use the slider to navigate.")
    logger.info("Dataset loaded successfully, returning updated UI components")

    return [
        dataset_info,  # dataset-info
        0,                   # min
        dataset_info['length'] - 1,  # max
        0,                # value
        marks,              # marks
        initial_message,    # datapoint-display
        create_dataset_info_display(dataset_info),  # dataset-info-display
        create_transforms_section(dataset_info['transforms']),  # transforms-section
    ]


@callback(
    outputs=Output('dataset-dropdown', 'options'),
    inputs=[Input('reload-button', 'n_clicks')],
    group="dataset"
)
def reload_datasets(n_clicks: Optional[int]) -> List[Dict[str, str]]:
    """Reload available datasets."""
    if n_clicks is None:
        raise PreventUpdate

    # Get list of available datasets
    available_datasets = registry.viewer.backend.get_available_datasets()

    # Create options for the dropdown
    return [{'label': label, 'value': name} for name, label in available_datasets.items()]
