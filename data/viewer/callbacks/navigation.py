"""Navigation-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.callbacks.display import create_display
from data.viewer.utils.settings_config import ViewerSettings

import logging
logger = logging.getLogger(__name__)




@callback(
    outputs=Output('datapoint-index-slider', 'value', allow_duplicate=True),
    inputs=[
        Input('prev-btn', 'n_clicks'),
        Input('next-btn', 'n_clicks'),
    ],
    states=[
        State('datapoint-index-slider', 'value'),
        State('datapoint-index-slider', 'min'),
        State('datapoint-index-slider', 'max'),
    ],
    group="navigation"
)
def update_index_from_buttons(
    prev_clicks: Optional[int],
    next_clicks: Optional[int],
    current_value: int,
    min_value: int,
    max_value: int
) -> List[int]:
    """Update the datapoint index based on prev/next button clicks."""
    if prev_clicks is None and next_clicks is None:
        raise PreventUpdate
    assert isinstance(current_value, int)

    # Get the ID of the button that triggered the callback
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Update value based on which button was clicked
    if triggered_id == 'prev-btn':
        new_value = max(min_value, current_value - 1)
    else:  # next-btn
        new_value = min(max_value, current_value + 1)

    return [new_value]


@callback(
    outputs=Output('current-index-display', 'children'),
    inputs=[Input('datapoint-index-slider', 'value')],
    group="navigation"
)
def update_current_index(current_idx: int) -> List[str]:
    """Update the current index display."""
    assert isinstance(current_idx, int)
    return [f"Current Index: {current_idx}"]


@callback(
    outputs=[
        Output('datapoint-display', 'children', allow_duplicate=True),
    ],
    inputs=[
        Input('datapoint-index-slider', 'value'),
        Input('3d-settings-store', 'data'),
        Input('camera-state', 'data')
    ],
    states=[
        State('dataset-info', 'data')
    ],
    group="navigation"
)
def update_datapoint_from_navigation(
    datapoint_idx: int,
    settings_3d: Optional[Dict[str, Union[str, int, float, bool]]],
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    camera_state: Dict
) -> List[html.Div]:
    """
    Update the displayed datapoint when navigation index changes.
    """
    logger.info(f"Navigation callback triggered - Index: {datapoint_idx}")

    if dataset_info is None or dataset_info == {} or 'name' not in dataset_info:
        logger.warning("No dataset info available for navigation")
        raise PreventUpdate

    dataset_name: str = dataset_info['name']
    logger.info(f"Navigating to index {datapoint_idx} in dataset: {dataset_name}")

    # For navigation updates, use all available transforms by default
    transforms = dataset_info.get('transforms', [])
    all_transform_indices = [transform['index'] for transform in transforms]

    # Get datapoint from backend through registry
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, all_transform_indices)

    # Get dataset type and create display
    dataset_type = dataset_info['type']  # Will raise KeyError if missing - that's a bug that should be caught
    
    logger.info(f"Dataset type: {dataset_type}")

    # Extract 3D settings and class labels using centralized configuration
    settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Call the display function with appropriate parameters
    logger.info(f"Creating {dataset_type} display for navigation")
    display = create_display(dataset_type, datapoint, class_labels, camera_state, settings)

    logger.info("Navigation display created successfully")
    return [display]
