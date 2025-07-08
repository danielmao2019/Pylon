"""3D settings-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
import json
from dash import Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback, registry
from data.viewer.utils.camera_utils import (
    update_figures_parallel,
    update_figure_camera,
    reset_figure_camera,
    get_default_camera_state
)
from data.viewer.utils.settings_config import ViewerSettings


@callback(
    outputs=[
        Output('3d-settings-store', 'data')
    ],
    inputs=[
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value'),
        Input('radius-slider', 'value'),
        Input('correspondence-radius-slider', 'value'),
        Input('lod-type-dropdown', 'value')
    ],
    group="3d_settings"
)
def update_3d_settings(
    point_size: float,
    point_opacity: float,
    sym_diff_radius: float,
    corr_radius: float,
    lod_type: str
) -> List[Dict[str, Union[str, int, float, bool]]]:
    """Update 3D settings store when slider values change.
    
    Args:
        point_size: Size of points in the 3D visualization
        point_opacity: Opacity of points (0.0 to 1.0)
        sym_diff_radius: Radius for symmetric difference computation
        corr_radius: Radius for correspondence visualization
        lod_type: Type of LOD to use ("continuous", "discrete", or "none")
    
    Returns:
        List containing a dictionary of all 3D settings
    """
    if point_size is None or point_opacity is None:
        raise PreventUpdate

    # Create settings dictionary with provided values
    raw_settings = {
        'point_size': point_size,
        'point_opacity': point_opacity,
        'sym_diff_radius': sym_diff_radius,
        'corr_radius': corr_radius,
        'lod_type': lod_type
    }
    
    # Validate and apply defaults using centralized configuration
    settings = ViewerSettings.validate_3d_settings(
        ViewerSettings.get_3d_settings_with_defaults(raw_settings)
    )

    # Update backend state with validated settings
    registry.viewer.backend.update_state(**settings)

    return [settings]


@callback(
    outputs=[
        Output('view-controls', 'style'),
        Output('pcr-controls', 'style')
    ],
    inputs=[Input('dataset-info', 'data')],
    group="3d_settings",
)
def update_view_controls(
    dataset_info: Optional[Dict[str, Union[str, int, bool, Dict]]]
) -> List[Dict[str, str]]:
    """Update the visibility of 3D view controls based on dataset type."""
    if dataset_info is None or not dataset_info:
        return [{'display': 'none'}, {'display': 'none'}]

    assert 'type' in dataset_info, f"{dataset_info.keys()=}"
    dataset_type = dataset_info.get('type')

    # Default styles
    view_controls_style = {'display': 'none'}
    pcr_controls_style = {'display': 'none'}

    # Show 3D controls for 3D datasets using proper API
    if ViewerSettings.requires_3d_visualization(dataset_type):
        view_controls_style = {'display': 'block'}

        # Show PCR controls only for PCR datasets
        if dataset_type == 'pcr':
            pcr_controls_style = {'display': 'block'}

    return [view_controls_style, pcr_controls_style]


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
    group="3d_settings"
)
def sync_camera_state(all_relayout_data: List[Dict[str, Any]], all_figures: List[Dict[str, Any]], _: Dict[str, Any]) -> List[Any]:
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
        State({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
    ],
    group="3d_settings"
)
def reset_camera_view(n_clicks: Optional[int], all_figures: List[Dict[str, Any]]) -> List[Any]:
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
