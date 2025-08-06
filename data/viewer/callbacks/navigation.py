"""Navigation-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, html, callback_context
import dash
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.callbacks.display import create_display
from data.viewer.utils.settings_config import ViewerSettings
from data.viewer.utils.debounce import debounce

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
@debounce
def update_index_from_buttons(
    prev_clicks: Optional[int],
    next_clicks: Optional[int],
    current_value: int,
    min_value: int,
    max_value: int
) -> List[int]:
    """Update the datapoint index based on prev/next button clicks."""
    if prev_clicks is None and next_clicks is None:
        # Thread-safe return instead of raising PreventUpdate in debounced context
        return [dash.no_update]
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
@debounce
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

    # Handle case where no dataset is selected (normal UI state)
    if dataset_info is None or dataset_info == {}:
        # Thread-safe return instead of raising PreventUpdate in debounced context
        return [dash.no_update]
    
    # Assert dataset info structure is valid - fail fast if corrupted
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
    settings_3d = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Get dataset instance for display method
    dataset_instance = registry.viewer.backend.get_dataset_instance(dataset_name=dataset_name)
    
    # All datasets must have display_datapoint method from base classes
    assert dataset_instance is not None, f"Dataset instance must not be None for dataset: {dataset_name}"
    assert hasattr(dataset_instance, 'display_datapoint'), f"Dataset {type(dataset_instance).__name__} must have display_datapoint method"
    
    display_func = dataset_instance.display_datapoint
    logger.info(f"Using display method from dataset class: {type(dataset_instance).__name__}")
    
    # Check if camera state is the default state - if so, pass None to allow camera pose calculation
    default_camera_state = {
        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5},
        'center': {'x': 0, 'y': 0, 'z': 0}, 
        'up': {'x': 0, 'y': 0, 'z': 1}
    }
    
    # For datasets that have camera poses (like iVISION MT), pass None to trigger pose calculation
    final_camera_state = camera_state
    if camera_state == default_camera_state:
        # Check if datapoint has camera pose - if so, let dataset calculate camera from pose
        if ('meta_info' in datapoint and 
            'camera_pose' in datapoint['meta_info'] and 
            hasattr(dataset_instance, '__class__') and 
            'iVISION' in dataset_instance.__class__.__name__):
            final_camera_state = None
            logger.info(f"Using None camera_state for {dataset_instance.__class__.__name__} to trigger camera pose calculation")
    
    # Create display using the determined display function
    display = create_display(
        display_func=display_func,
        datapoint=datapoint,
        class_labels=class_labels,
        camera_state=final_camera_state,
        settings_3d=settings_3d
    )

    logger.info("Navigation display created successfully")
    return [display]
