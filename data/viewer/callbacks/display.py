"""Display-related callbacks for the viewer."""
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import logging
from typing import Dict, List, Optional, Union, Literal
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.callbacks.registry import callback, registry


logger = logging.getLogger(__name__)

# Dataset type definitions
DatasetType = Literal['2d_change_detection', '3d_change_detection', 'point_cloud_registration']

# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    '2d_change_detection': display_2d_datapoint,
    '3d_change_detection': display_3d_datapoint,
    'point_cloud_registration': display_pcr_datapoint
}

@callback(
    outputs=[
        Output('camera-state', 'data'),
    ],
    inputs=[
        Input({'type': 'point-cloud-graph', 'index': 0}, 'relayoutData'),
        Input({'type': 'point-cloud-graph', 'index': 1}, 'relayoutData'),
        Input({'type': 'point-cloud-graph', 'index': 2}, 'relayoutData'),
        Input({'type': 'point-cloud-graph', 'index': 3}, 'relayoutData'),
        State('camera-state', 'data'),
    ],
    group="display"
)
def update_camera_state(relayout_data_0, relayout_data_1, relayout_data_2, relayout_data_3, current_camera_state):
    """Update the camera state when any point cloud view is manipulated."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Get the relayout_data that triggered the callback
    relayout_data = None
    for trigger in ctx.triggered:
        if '.relayoutData' in trigger['prop_id']:
            relayout_data = trigger['value']
            break

    if not relayout_data:
        raise PreventUpdate

    # Check if the relayout_data contains camera information
    camera_keys = ['scene.camera']
    if not any(key in relayout_data for key in camera_keys):
        raise PreventUpdate

    # Extract camera state from relayout_data
    new_camera_state = {}
    if 'scene.camera' in relayout_data:
        camera_data = relayout_data['scene.camera']
        new_camera_state = {
            'up': camera_data.get('up', current_camera_state.get('up')),
            'center': camera_data.get('center', current_camera_state.get('center')),
            'eye': camera_data.get('eye', current_camera_state.get('eye'))
        }

    return [new_camera_state]

@callback(
    outputs=[
        Output('datapoint-display', 'children'),
    ],
    inputs=[
        Input('dataset-info', 'data'),
        Input('datapoint-index-slider', 'value'),
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value'),
        Input('radius-slider', 'value'),
        Input('camera-state', 'data')
    ],
    group="display"
)
def update_datapoint(
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]],
    datapoint_idx: int,
    point_size: float,
    point_opacity: float,
    radius: float,
    camera_state: Dict
) -> List[html.Div]:
    """
    Update the displayed datapoint based on the slider value.
    Also handles 3D point cloud visualization settings.
    """
    logger.info(f"Display update callback triggered - Dataset info: {dataset_info}, Index: {datapoint_idx}")
    
    if dataset_info is None or dataset_info == {}:
        logger.warning("No dataset info available")
        return [html.Div("No dataset loaded.")]

    dataset_name: str = dataset_info.get('name', 'unknown')
    logger.info(f"Attempting to get dataset: {dataset_name}")
    
    # Get datapoint from manager through registry
    datapoint = registry.viewer.dataset_manager.get_datapoint(dataset_name, datapoint_idx)

    # Get dataset type and is_3d from dataset info
    dataset_type: str = dataset_info.get('type', 'change_detection')
    is_3d: bool = dataset_info.get('is_3d', False)
    logger.info(f"Dataset type: {dataset_type}, 3D: {is_3d}")

    # Get class labels if available
    class_labels: Dict[int, str] = dataset_info.get('class_labels', {})
    logger.info(f"Class labels available: {bool(class_labels)}")

    # Determine the appropriate display function based on dataset type
    if dataset_type == 'point_cloud_registration':
        display_type = 'point_cloud_registration'
    elif is_3d:
        display_type = '3d_change_detection'
    else:
        display_type = '2d_change_detection'
        
    # Get the appropriate display function
    display_func = DISPLAY_FUNCTIONS.get(display_type)
    if display_func is None:
        logger.error(f"No display function found for dataset type: {display_type}")
        return [html.Div(f"Error: Unsupported dataset type: {display_type}")]
        
    # Call the display function with appropriate parameters
    logger.info(f"Creating {display_type} display")
    if display_type == 'point_cloud_registration':
        display = display_func(datapoint, point_size, point_opacity, camera_state, radius)
    elif display_type == '3d_change_detection':
        display = display_func(datapoint, point_size, point_opacity, class_labels, camera_state)
    else:  # 2d_change_detection
        display = display_func(datapoint)
        
    logger.info("Display created successfully")
    return [display]
