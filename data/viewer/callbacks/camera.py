"""Camera pose and synchronization callbacks for the viewer."""
from typing import Dict, List, Optional, Any, Tuple
import json
from dash import Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback
from data.viewer.utils.camera_utils import (
    update_figures_parallel,
    update_figure_camera,
    reset_figure_camera,
    get_default_camera_state
)
from data.viewer.utils.settings_config import ViewerSettings


@callback(
    outputs=[
        Output({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
        Output('camera-state', 'data'),
    ],
    inputs=[
        Input({'type': 'point-cloud-graph', 'index': ALL}, 'relayoutData'),
    ],
    states=[
        State({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
    ],
    group="camera"
)
def sync_camera_state(all_relayout_data: List[Dict[str, Any]], all_figures: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synchronize camera state across all point cloud views when user drags/interacts with 3D graphs."""
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

    # Update all figures with the new camera state using centralized utility
    update_func = update_figure_camera(triggered_index, new_camera)
    updated_figures = update_figures_parallel(
        all_figures, 
        update_func, 
        ViewerSettings.PERFORMANCE_SETTINGS['max_thread_workers']
    )

    return updated_figures, new_camera


@callback(
    outputs=[
        Output({'type': 'point-cloud-graph', 'index': ALL}, 'figure', allow_duplicate=True),
        Output('camera-state', 'data', allow_duplicate=True),
    ],
    inputs=[
        Input('reset-camera-button', 'n_clicks'),
    ],
    states=[
        State({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
    ],
    group="camera"
)
def reset_camera_view(n_clicks: Optional[int], all_figures: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reset camera view to default position for all point cloud graphs."""
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    # Get default camera state from centralized configuration
    default_camera = get_default_camera_state()

    # Update all figures with default camera state using centralized utility
    reset_func = reset_figure_camera(default_camera)
    updated_figures = update_figures_parallel(
        all_figures, 
        reset_func, 
        ViewerSettings.PERFORMANCE_SETTINGS['max_thread_workers']
    )

    return updated_figures, default_camera

