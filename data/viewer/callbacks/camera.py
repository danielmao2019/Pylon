"""Camera pose and synchronization callbacks for the viewer."""
from typing import Dict, List, Optional, Any, Tuple
import json
from dash import Input, Output, State, ctx, ALL
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback
from data.viewer.utils.camera_utils import (
    update_figures_parallel,
    update_figure_camera,
    reset_figure_camera,
    get_default_camera_state
)
from data.viewer.utils.settings_config import ViewerSettings
from data.viewer.utils.debounce import debounce


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
@debounce
def sync_camera_state(all_relayout_data: List[Dict[str, Any]], all_figures: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synchronize camera state across all point cloud views when user drags/interacts with 3D graphs."""
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(all_relayout_data, list), f"all_relayout_data must be list, got {type(all_relayout_data)}"
    assert isinstance(all_figures, list), f"all_figures must be list, got {type(all_figures)}"
    assert len(all_relayout_data) == len(all_figures), f"Lists must have same length: {len(all_relayout_data)} vs {len(all_figures)}"
    
    # Since we're in a debounced (threaded) context, we can't access ctx.triggered
    # Instead, we determine which graph was updated by checking for non-None relayoutData
    triggered_index = None
    triggered_relayout_data = None
    
    for i, relayout_data in enumerate(all_relayout_data):
        if relayout_data is not None and isinstance(relayout_data, dict) and len(relayout_data) > 0:
            # Skip if it's just a timestamp or other non-camera update
            camera_keys = ['scene.camera', 'scene.camera.center', 'scene.camera.eye', 'scene.camera.up']
            if any(key in str(relayout_data) for key in camera_keys):
                triggered_index = i
                triggered_relayout_data = relayout_data
                break
    
    if triggered_index is None or triggered_relayout_data is None:
        raise PreventUpdate

    # Use the triggered relayout data we already found
    relayout_data = triggered_relayout_data
    
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
