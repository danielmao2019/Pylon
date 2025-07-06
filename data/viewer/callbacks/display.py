"""Display-related callbacks for the viewer."""
from typing import Dict, List, Optional, Union, Any
import json
from dash import Input, Output, State, callback_context, html, ALL
from dash.exceptions import PreventUpdate
import dash
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.viewer.callbacks.registry import callback, registry
from data.viewer.layout.controls.transforms import create_transforms_section
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

    # Update all figures with the new camera state
    def update_figure_camera(i, figure):
        if not figure:
            # Skip empty figures
            return dash.no_update
        
        # Always update all figures including the triggered one to ensure consistency
        # Deep copy the figure to avoid reference issues
        import copy
        updated_figure = copy.deepcopy(figure)
        
        # Ensure layout structure exists
        if 'layout' not in updated_figure:
            updated_figure['layout'] = {}
        if 'scene' not in updated_figure['layout']:
            updated_figure['layout']['scene'] = {}
            
        # Update camera for all figures
        updated_figure['layout']['scene']['camera'] = new_camera
        
        # Preserve other scene properties to avoid display mixing
        if 'scene' in figure.get('layout', {}):
            for key, value in figure['layout']['scene'].items():
                if key != 'camera':
                    updated_figure['layout']['scene'][key] = value
        
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
    group="display"
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


@callback(
    outputs=[
        Output('datapoint-display', 'children'),
    ],
    inputs=[
        Input('dataset-info', 'data'),
        Input('datapoint-index-slider', 'value'),
        Input({'type': 'transform-checkbox', 'index': ALL}, 'value'),
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
    transform_values: List[List[int]],
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

    # Get list of selected transform indices
    selected_indices = [
        idx for values in transform_values
        for idx in values  # values will be a list containing the index if checked, empty if not
    ]

    # Get datapoint from backend through registry
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, selected_indices)

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


@callback(
    outputs=[
        Output('transforms-section', 'children', allow_duplicate=True),
        Output('datapoint-display', 'children', allow_duplicate=True)
    ],
    inputs=[Input('dataset-info', 'data')],
    states=[State('datapoint-index-slider', 'value')],
    group="display"
)
def update_transforms(dataset_info: Dict[str, Any], datapoint_idx: Optional[int]) -> List[Union[html.Div, List[html.Div]]]:
    """Update the transforms section when dataset info changes and display datapoint with all transforms applied."""
    # If no dataset is selected, maintain current state
    if not dataset_info:
        raise PreventUpdate

    dataset_name = dataset_info['name']
    transforms = dataset_info.get('transforms', [])

    # Use index 0 if datapoint_idx is None (first dataset selection)
    if datapoint_idx is None:
        datapoint_idx = 0

    # Create updated transforms section
    transforms_section = create_transforms_section(transforms)

    # Display datapoint with all transforms applied by default
    # Get all transform indices to apply all transforms by default
    all_transform_indices = [transform['index'] for transform in transforms]

    # Get transformed datapoint using backend
    datapoint = registry.viewer.backend.get_datapoint(dataset_name, datapoint_idx, all_transform_indices)

    # Get dataset type and determine display function
    dataset_type = dataset_info['type']

    # Get the appropriate display function
    display_func = DISPLAY_FUNCTIONS.get(dataset_type)
    if display_func is None:
        return [transforms_section, [html.Div(f"Error: Unsupported dataset type: {dataset_type}")]]

    # Call the display function with appropriate parameters (using default 3D settings)
    if dataset_type == 'semseg':
        display = display_func(datapoint)
    elif dataset_type == '2dcd':
        display = display_func(datapoint)
    elif dataset_type == '3dcd':
        display = display_func(datapoint, 1.0, 1.0, dataset_info.get('class_labels', {}), {})
    elif dataset_type == 'pcr':
        display = display_func(datapoint, 1.0, 1.0, {}, 1.0)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return [transforms_section, [display]]
