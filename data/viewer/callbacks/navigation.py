"""Navigation-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.layout.display.display_semseg import display_semseg_datapoint
from data.viewer.utils.settings_config import ViewerSettings

import logging
logger = logging.getLogger(__name__)


# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    'semseg': display_semseg_datapoint,
    '2dcd': display_2dcd_datapoint,
    '3dcd': display_3dcd_datapoint,
    'pcr': display_pcr_datapoint,
}


def _create_display(
    dataset_type: str,
    display_func: Any,
    datapoint: Dict[str, Any], 
    class_labels: Optional[Dict[int, str]],
    camera_state: Dict[str, Any],
    settings_3d: Dict[str, Union[float, str]]
) -> html.Div:
    """Create display based on dataset type with appropriate parameters."""
    # Define parameter mappings for each dataset type
    display_params = {
        'semseg': {
            'args': (datapoint,),
            'kwargs': {}
        },
        '2dcd': {
            'args': (datapoint,),
            'kwargs': {}
        },
        '3dcd': {
            'args': (datapoint, class_labels, camera_state),
            'kwargs': {
                'point_size': settings_3d['point_size'],
                'point_opacity': settings_3d['point_opacity'],
                'lod_type': settings_3d['lod_type']
            }
        },
        'pcr': {
            'args': (),
            'kwargs': {
                'datapoint': datapoint,
                'camera_state': camera_state,
                'point_size': settings_3d['point_size'],
                'point_opacity': settings_3d['point_opacity'],
                'sym_diff_radius': settings_3d['sym_diff_radius'],
                'corr_radius': settings_3d['corr_radius'],
                'lod_type': settings_3d['lod_type']
            }
        }
    }
    
    if dataset_type not in display_params:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    params = display_params[dataset_type]
    return display_func(*params['args'], **params['kwargs'])


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
    ],
    states=[
        State('dataset-info', 'data'),
        State('camera-state', 'data'),
        State('3d-settings-store', 'data')
    ],
    group="navigation"
)
def update_datapoint_from_navigation(
    datapoint_idx: int,
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    camera_state: Dict,
    settings_3d: Optional[Dict[str, Union[str, int, float, bool]]]
) -> List[html.Div]:
    """
    Update the displayed datapoint when navigation index changes.
    """
    logger.info(f"Navigation callback triggered - Index: {datapoint_idx}")

    if dataset_info is None or dataset_info == {}:
        logger.warning("No dataset info available for navigation")
        return [html.Div("No dataset loaded.")]

    dataset_name: str = dataset_info.get('name', 'unknown')
    logger.info(f"Navigating to index {datapoint_idx} in dataset: {dataset_name}")

    # For navigation updates, use all available transforms by default
    transforms = dataset_info.get('transforms', [])
    all_transform_indices = [transform['index'] for transform in transforms]

    # Get datapoint from backend through registry
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, all_transform_indices)

    # Get dataset type and display function
    dataset_type = dataset_info['type']  # Will raise KeyError if missing - that's a bug that should be caught
    display_func = DISPLAY_FUNCTIONS[dataset_type]  # Will raise KeyError if unsupported - that's a bug that should be caught
    
    logger.info(f"Dataset type: {dataset_type}")

    # Extract 3D settings and class labels using centralized configuration
    settings = ViewerSettings.get_3d_settings_with_defaults(settings_3d)
    class_labels = dataset_info.get('class_labels') if dataset_type in ['semseg', '3dcd'] else None

    # Call the display function with appropriate parameters
    logger.info(f"Creating {dataset_type} display for navigation")
    display = _create_display(dataset_type, display_func, datapoint, class_labels, camera_state, settings)

    logger.info("Navigation display created successfully")
    return [display]
