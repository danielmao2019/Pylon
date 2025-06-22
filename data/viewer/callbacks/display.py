"""Display-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
import json
from dash import Input, Output, State, callback_context, html, ALL
from dash.exceptions import PreventUpdate
import dash
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.layout.display.display_semseg import display_semseg_datapoint

import logging
logger = logging.getLogger(__name__)


# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    'semseg': display_semseg_datapoint,
    '2dcd': display_2dcd_datapoint,
    '3dcd': display_3dcd_datapoint,
    'pcr': display_pcr_datapoint,
}

@callback(
    outputs=[
        Output({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
        Output('camera-state', 'data'),
    ],
    inputs=[
        Input({'type': 'point-cloud-graph', 'index': ALL}, 'relayoutData'),
        State({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
        State('camera-state', 'data'),
    ],
    group="display"
)
def sync_camera_state(all_relayout_data: List[Dict[str, Any]], all_figures: List[Dict[str, Any]], current_camera_state: Dict[str, Any]) -> List[Any]:
    """Synchronize camera state across all point cloud views."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Find which graph triggered the update
    triggered_prop_id = ctx.triggered[0]['prop_id']
    if 'relayoutData' not in triggered_prop_id:
        raise PreventUpdate

    # Parse the triggered component ID to get the index
    try:
        triggered_id = json.loads(triggered_prop_id.split('.')[0])
        triggered_index = triggered_id['index']
    except (json.JSONDecodeError, KeyError, IndexError):
        raise PreventUpdate

    # Get the relayout data from the triggered graph
    if triggered_index >= len(all_relayout_data) or not all_relayout_data[triggered_index]:
        raise PreventUpdate

    relayout_data = all_relayout_data[triggered_index]

    # Check if the relayout_data contains camera information
    if 'scene.camera' not in relayout_data:
        raise PreventUpdate

    # Extract new camera state
    new_camera = relayout_data['scene.camera']
    
    # Update all figures with the new camera state, except the one that triggered the change
    updated_figures = []
    for i, figure in enumerate(all_figures):
        if i == triggered_index or not figure:
            # Don't update the triggering graph or empty figures
            updated_figures.append(dash.no_update)
        else:
            # Create updated figure with new camera state
            updated_figure = figure.copy()
            if 'layout' not in updated_figure:
                updated_figure['layout'] = {}
            if 'scene' not in updated_figure['layout']:
                updated_figure['layout']['scene'] = {}
            updated_figure['layout']['scene']['camera'] = new_camera
            updated_figures.append(updated_figure)

    return updated_figures, new_camera

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

    # Get dataset type from dataset info
    dataset_type = dataset_info.get('type')
    if dataset_type is None:
        raise ValueError("Dataset type not available")

    logger.info(f"Dataset type: {dataset_type}")

    # Get class labels if available
    if dataset_type in ['semseg', '3dcd']:
        assert 'class_labels' in dataset_info, f"{dataset_info.keys()=}"
        class_labels: Dict[int, str] = dataset_info['class_labels']

    # Get the appropriate display function
    display_func = DISPLAY_FUNCTIONS.get(dataset_type)
    if display_func is None:
        logger.error(f"No display function found for dataset type: {dataset_type}")
        return [html.Div(f"Error: Unsupported dataset type: {dataset_type}")]

    # Call the display function with appropriate parameters
    logger.info(f"Creating {dataset_type} display")
    if dataset_type == 'semseg':
        display = display_func(datapoint)
    elif dataset_type == '2dcd':
        display = display_func(datapoint)
    elif dataset_type == '3dcd':
        display = display_func(datapoint, point_size, point_opacity, class_labels, camera_state)
    elif dataset_type == 'pcr':
        display = display_func(datapoint, point_size, point_opacity, camera_state, radius)
    else:
        assert 0, f"{dataset_type=}"

    logger.info("Display created successfully")
    return [display]
