"""3D settings-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
import json
from dash import Input, Output, State, callback_context, ALL
from dash.exceptions import PreventUpdate
import dash
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.viewer.callbacks.registry import callback, registry


# Dataset types that use 3D visualization
THREE_D_DATASET_TYPES = ['3dcd', 'pcr']


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

    # Update backend state with new 3D settings
    registry.viewer.backend.update_state(
        point_size=point_size,
        point_opacity=point_opacity,
        sym_diff_radius=sym_diff_radius or 0.05,
        corr_radius=corr_radius or 0.1,
        lod_type=lod_type or "continuous"
    )

    # Store all 3D settings in the store
    settings = {
        'point_size': point_size,
        'point_opacity': point_opacity,
        'sym_diff_radius': sym_diff_radius or 0.05,  # Default symmetric difference radius
        'corr_radius': corr_radius or 0.1,  # Default correspondence radius
        'lod_type': lod_type or "continuous"  # Default to continuous LOD
    }

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

    # Show 3D controls for 3D datasets
    if dataset_type in THREE_D_DATASET_TYPES:
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

    # Update all figures with the new camera state in parallel
    def update_figure_camera(i, figure):
        if i == triggered_index or not figure:
            # Don't update the triggering graph or empty figures
            return dash.no_update
        else:
            # Create updated figure with new camera state
            updated_figure = figure.copy()
            if 'layout' not in updated_figure:
                updated_figure['layout'] = {}
            if 'scene' not in updated_figure['layout']:
                updated_figure['layout']['scene'] = {}
            updated_figure['layout']['scene']['camera'] = new_camera
            return updated_figure

    updated_figures = [None] * len(all_figures)
    
    # Use parallel processing for multiple figures
    if len(all_figures) > 1:
        with ThreadPoolExecutor(max_workers=min(len(all_figures), 4)) as executor:
            # Submit all update tasks
            future_to_index = {
                executor.submit(update_figure_camera, i, figure): i 
                for i, figure in enumerate(all_figures)
            }
            
            # Collect results in order
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                updated_figures[idx] = future.result()
    else:
        # For single figure, just update directly
        updated_figures = [update_figure_camera(i, figure) for i, figure in enumerate(all_figures)]

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

    # Default camera state
    default_camera = {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
    }

    # Update all figures with default camera state in parallel
    def reset_figure_camera(figure):
        if not figure:
            return dash.no_update

        updated_figure = figure.copy()
        if 'layout' not in updated_figure:
            updated_figure['layout'] = {}
        if 'scene' not in updated_figure['layout']:
            updated_figure['layout']['scene'] = {}
        updated_figure['layout']['scene']['camera'] = default_camera
        return updated_figure

    updated_figures = [None] * len(all_figures)
    
    # Use parallel processing for multiple figures
    if len(all_figures) > 1:
        with ThreadPoolExecutor(max_workers=min(len(all_figures), 4)) as executor:
            # Submit all reset tasks
            future_to_index = {
                executor.submit(reset_figure_camera, figure): i 
                for i, figure in enumerate(all_figures)
            }
            
            # Collect results in order
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                updated_figures[idx] = future.result()
    else:
        # For single figure, just update directly
        updated_figures = [reset_figure_camera(figure) for figure in all_figures]

    return updated_figures, default_camera
