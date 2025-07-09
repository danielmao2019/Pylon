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
    camera_state: Dict,
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]]
) -> List[html.Div]:
    """
    Update the displayed datapoint when navigation index changes.
    """
    logger.info(f"Navigation callback triggered - Index: {datapoint_idx}")

    # Assert dataset info is valid - fail fast if not
    assert dataset_info is not None, "Dataset info must not be None"
    assert dataset_info != {}, "Dataset info must not be empty"
    assert 'name' in dataset_info, f"Dataset info must have 'name' key, got keys: {list(dataset_info.keys())}"
    assert 'type' in dataset_info, f"Dataset info must have 'type' key, got keys: {list(dataset_info.keys())}"
    assert 'transforms' in dataset_info, f"Dataset info must have 'transforms' key, got keys: {list(dataset_info.keys())}"

    dataset_name: str = dataset_info['name']
    dataset_type: str = dataset_info['type']
    logger.info(f"Navigating to index {datapoint_idx} in dataset: {dataset_name}")

    # For navigation updates, use all available transforms by default
    transforms = dataset_info['transforms']
    all_transform_indices = [transform['index'] for transform in transforms]

    # Get datapoint from backend through registry using kwargs
    datapoint = registry.viewer.backend.get_datapoint(
        dataset_name=dataset_name,
        index=datapoint_idx,
        transform_indices=all_transform_indices
    )

    logger.info(f"Dataset type: {dataset_type}")

    # Extract 3D settings and class labels using centralized configuration
    settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Create display using kwargs to prevent parameter ordering issues
    display = create_display(
        dataset_type=dataset_type,
        datapoint=datapoint,
        class_labels=class_labels,
        camera_state=camera_state,
        settings=settings
    )

    logger.info("Navigation display created successfully")
    return [display]
